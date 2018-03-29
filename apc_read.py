"""Amazon Picking Challenge dataset reader."""
import json
import os
import pickle
import yaml

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import skimage.data
import skimage.color
import skimage.feature
import skimage.measure
import skimage.transform


class Struct:
    """A convenient struct-like class.

    Source: Python Cookbook, Beazley and Jones. 3rd Edition. 8.11. Simplifying
        the Initialization of Data Structures.

    Usage:
        ```python
        class Stock(Structure):
            _fields = ['name', 'shares', 'price']

        s1 = Stock('ACME', 50, 91.1)
        s2 = Stock('ACME', 50, price=91.1)
        s3 = Stock('ACME', shares=50, price=91.1)
        ```
    """
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # Set all of the positional arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # Set the remaining keyword arguments
        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))

        # Check for any remaining unknown arguments
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))


class Error(Struct):
    _fields = ['rotation', 'translation']


class ObjView(Struct):
    _fields = ['depth',
               'descriptors',
               'img',
               'inlier_points',
               'keypoints',
               'pose']


def _detect_and_extract(img, descriptor_extractor):
    """Extracts features from `img` and returns the resulting keypoints and
    descriptors.
    """
    descriptor_extractor.detect_and_extract(img)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors

    return keypoints, descriptors


def _get_pose(fname):
    """Reads a raw file corresponding to `fname`, and returns the pose stored
    in that file.
    """
    with open(fname, 'r') as f:
        pose_raw = f.read()

    return yaml.load(pose_raw.replace('!!opencv-matrix', '')[10:])


def _triangulation_loss_depth(world_coords, screen_coords, views, depth_coef):
    """Compute triangulation loss with known depth."""
    focal_length = 525
    world_coords = world_coords.reshape([screen_coords.shape[0], 2])

    mse = 0.0
    for i, view in enumerate(views):
        rotation = view.pose['object_rotation_wrt_camera']['data']
        translation = view.pose['object_translation_wrt_camera']

        r0 = rotation[:1*3]
        r1 = rotation[1*3:2*3]
        r2 = rotation[2*3:]

        t_x = translation[0]
        t_y = translation[1]
        t_z = translation[2]

        # NOTE(brendan): The values to estimate.
        x_c = world_coords[:, 0]
        y_c = world_coords[:, 1]

        pixel = screen_coords[:, i, :].astype(np.int32)
        z_c = np.array([view.depth[p[1] + 240, p[0] + 320] for p in pixel])
        z_c *= depth_coef

        # r_{20}*x_c + r_{21}*y_c + r_{22}*z_c + t_z
        denominator = (r2[0]*x_c + r2[1]*y_c + r2[2]*z_c + t_z)

        # denominator*(x_s, y_s)
        target = denominator[:, np.newaxis] * screen_coords[:, i, :]

        # r_{00}*x_c + r_{01}*y_c + r_{02}*z_c + t_x
        prediction_x = focal_length*(r0[0]*x_c + r0[1]*y_c + r0[2]*z_c + t_x)
        prediction_y = focal_length*(r1[0]*x_c + r1[1]*y_c + r1[2]*z_c + t_y)
        prediction = np.concatenate(
            [prediction_x[:, np.newaxis], prediction_y[:, np.newaxis]], axis=1)

        error = target - prediction
        error = error.reshape(-1)
        mse += np.dot(error, error)

    mse /= len(views)

    return mse


def _triangulation_loss(params, screen_coords, views):
    """Minimize the residuals by directly reprojecting the estimated world
    coordinates of a point, and comparing against the known screen coordinates
    of that point.

    Args:
        params: A parameter for converting depth as measured by the Kinect-v1
            sensor to depth in meters, concatenated to a matrix of shape (N, 2)
            of N points (x_c, y_c), each of which is an estimate of the world
            coordinates for the given point. Units: meters.
        screen_coords: A matrix of shape (N, M, 2), where N is the number of data
            points corresponding to the given keypoint the world coordinates of
            which are being estimated, M is the number of poses, and there are
            two coordinates (x_s, y_s) for each pose and data point
            combination. Units: pixels.
        views: The two views, containing (among other information) a list of
            length M of the pose labels from the APC dataset. Pose rotations
            are unitless values (?), and translations are in meters.

    Returns:
        The MSE between x_s and the re-projected x_s'.
    """
    # NOTE(brendan): The focal length is 525px. Source:
    # http://wiki.ros.org/kinect_calibration/technical#Lens_distortion_and_focal_length
    depth_coef = params[0]
    world_coords = params[1:]

    return _triangulation_loss_depth(world_coords,
                                     screen_coords,
                                     views,
                                     depth_coef)


def _get_view(objname, descriptor_extractor, cam, shelf):
    """Construct a view for `objname` and camera view `cam`."""
    view = ObjView(
        depth=None,
        descriptors=None,
        img=skimage.io.imread(f'{objname}-image-{shelf}-1-{cam}-0.png'),
        inlier_points=None,
        keypoints=None,
        pose=_get_pose(f'{objname}-pose-{shelf}-1-{cam}-0.yml'))

    view.depth = []
    for i in range(4):
        view.depth.append(
            skimage.io.imread(f'{objname}-depth-{shelf}-1-{cam}-{i}.png'))

    # NOTE(brendan): We stack the depth maps and take the median here, to throw
    # away outliers and get a reasonable reading from the depth sensor.
    view.depth = np.stack(view.depth)
    view.depth = np.median(view.depth, axis=0)

    view.img = skimage.color.rgb2gray(view.img)
    view.keypoints, view.descriptors = _detect_and_extract(
        view.img, descriptor_extractor)

    return view


def _match_discard_outliers(view1, view2):
    """Match the descriptors for view1 and view2, and return all the matches
    along with the inliers.
    """
    matches = skimage.feature.match_descriptors(view1.descriptors,
                                                view2.descriptors,
                                                cross_check=True)

    data = (view1.keypoints[matches[:, 0]], view2.keypoints[matches[:, 1]])
    _, inliers = skimage.measure.ransac(
        data,
        model_class=skimage.transform.FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=1,
        max_trials=5000)

    return matches, inliers


def _rotation_mtx_from_quaternions(q):
    """Compute rotation matrix R from quaternions q."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    r0 = [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)]
    r0 = np.array(r0)

    r1 = [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)]
    r1 = np.array(r1)

    r2 = [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    r2 = np.array(r2)

    return r0, r1, r2


def _pose_estimation_loss(params,
                          screen_coords,
                          world_coords,
                          view,
                          depth_coef):
    """Compute pose estimation loss for optimizer inner loop.

    Args:
        params: Seven camera parameters.
        screen_coords: (N, 2) coordinates of keypoints on image sensor.
        world_coords: (N, 2) 3D X, Y coordinates.
        view: The view to estimate pose for.
        depth_coef: Converts from the Kinect-v1 sensor depth to meters.

    Returns:
        MSE mean squared error of predicted (re-projected) screen coordinates
        with target.
    """
    focal_length = 525

    # NOTE(brendan): The values to estimate.
    q = params[:4]
    r0, r1, r2 = _rotation_mtx_from_quaternions(q)

    t_x = params[4 + 0]
    t_y = params[4 + 1]
    t_z = params[4 + 2]

    world_coords = np.stack(world_coords)
    x_c = world_coords[:, 0]
    y_c = world_coords[:, 1]

    pixel = screen_coords.astype(np.int32)
    z_c = np.array([view.depth[p[1] + 240, p[0] + 320] for p in pixel])
    z_c *= depth_coef

    # r_{20}*x_c + r_{21}*y_c + r_{22}*z_c + t_z
    denominator = (r2[0]*x_c + r2[1]*y_c + r2[2]*z_c + t_z)

    # denominator*(x_s, y_s)
    target = denominator[:, np.newaxis] * screen_coords

    # r_{00}*x_c + r_{01}*y_c + r_{02}*z_c + t_x
    prediction_x = focal_length*(r0[0]*x_c + r0[1]*y_c + r0[2]*z_c + t_x)
    prediction_y = focal_length*(r1[0]*x_c + r1[1]*y_c + r1[2]*z_c + t_y)
    prediction = np.concatenate(
        [prediction_x[:, np.newaxis], prediction_y[:, np.newaxis]], axis=1)

    error = target - prediction
    error = error.reshape(-1)
    mse = np.dot(error, error)

    mse += np.abs(np.sum(q**2) - 1)

    return mse


def _compute_pose_error(test_view,
                        world_coords_test,
                        screen_coords_test,
                        depth_coef):
    """Use `cv2.solvePnPRansac` (or my own function if it ever works...) to estimate pose."""
    object_points = np.zeros([len(world_coords_test), 3])
    for i, world_coord in enumerate(world_coords_test):
        object_points[i, 0] = world_coord[0]
        object_points[i, 1] = world_coord[1]
        pixel = screen_coords_test.astype(np.int32)
        object_points[i, 2] = depth_coef*test_view.depth[pixel[i, 1] + 240,
                                                         pixel[i, 0] + 320]
    camera_mtx = np.array([[525, 0, 0], [0, 525, 0], [0, 0, 1]],
                          dtype=np.float64)
    _, pred_rot, pred_trans, inliers_pnp = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=screen_coords_test,
        cameraMatrix=camera_mtx,
        distCoeffs=None)
    if inliers_pnp is None:
        print()
        return None
    pred_trans = pred_trans.squeeze()
    pred_rot = pred_rot.squeeze()

    translation_error = (pred_trans -
                         test_view.pose['object_translation_wrt_camera'])
    translation_error = np.linalg.norm(translation_error)
    print(f'translation error: {translation_error} meters')

    test_rotation = test_view.pose['object_rotation_wrt_camera']['data']
    test_rotation = np.array(test_rotation).reshape(3, 3)
    test_rotation = cv2.Rodrigues(test_rotation)[0].squeeze()
    rotation_error = np.linalg.norm(pred_rot - test_rotation)
    print(f'rotation error: {rotation_error} radians')


    # TODO(brendan): Gosh, why doesn't this work?
    # Switch to axis/angle rotation representation?
    # NOTE(brendan): Four quaternion parameters representing the rotation,
    # three translation parameters.
    # initial_guesses = np.zeros(7) + 0.5
    # args = (screen_coords_test[:, 0, :],
    #         world_coords_test,
    #         test_view,
    #         depth_coef)
    # optimize_result = _solve(initial_guesses,
    #                          _pose_estimation_loss,
    #                          bounds=(-np.inf, np.inf),
    #                          args=args)
    # print(_rotation_mtx_from_quaternions(initial_guesses[:4]))

    print()

    return rotation_error, translation_error


def _predict_pose(objname, descriptor_extractor, test_view):
    """Predicts the pose of a third image, from the first two."""
    max_matches = 0
    min_error = Error(rotation=np.inf, translation=np.inf)
    for shelf in ['B', 'C', 'E', 'F', 'H', 'I', 'K', 'L']:
        train_view = _get_view(objname, descriptor_extractor, cam=1, shelf=shelf)

        directory = f'features/{objname}/{shelf}'
        with open(f'{directory}/f.pkl', 'rb') as f:
            saved_matches = pickle.load(f)

        depth_coef = saved_matches['depth_coef']
        inliers_train = saved_matches['inliers']
        matches_train = saved_matches['matches']
        train_view = saved_matches['view1']
        world_coords_train = saved_matches['world_coords']

        matches_test, inliers_test = _match_discard_outliers(test_view,
                                                             train_view)

        matches_test = [m for m in matches_test[inliers_test]
                        if m[1] in matches_train[inliers_train, 0]]
        print(f'common matches: {len(matches_test)}')
        # NOTE(brendan): Only bother to run the predictions if there are at
        # least 8 points.
        if len(matches_test) < 8:
            print()
            continue
        matches_test = np.array(matches_test)

        print(f'train_view inliers: {len(train_view.inlier_points)}')

        test_view.inlier_points = test_view.keypoints[matches_test[:, 0]]
        train_view.inlier_points = train_view.keypoints[matches_test[:, 1]]
        # NOTE(brendan): We have saved a set of world coordinates in
        # `world_coords`, where the list corresponds to the same sequence of
        # screen coordinates in `screen_coords_train`, for train_view (view 1)
        # and view 2.
        #
        # Here, we want to take all of those screen coordinates for view 1 that
        # have a feature that matches up with a feature in test_view, and save
        # all the world coordinates corresponding to those features.
        #
        # Hence, `screen_coords_train[i, 0, :]` where i corresponds to
        # world_coords_train[i] should be the same as
        # `screen_coords_test[j, 1, :]` where j corresponds to
        # world_coords_test[j]
        match_train_to_world_coord = {}
        for i, m in enumerate(matches_train[inliers_train, 0]):
            match_train_to_world_coord.update({m: world_coords_train[i]})

        screen_coords_test = _get_screen_coords(test_view, train_view)

        world_coords_test = []
        for m in matches_test[:, 1]:
            world_coords_test.append(match_train_to_world_coord[m])

        errors = _compute_pose_error(test_view,
                                     world_coords_test,
                                     screen_coords_test[:, 0, :],
                                     depth_coef)
        if errors is None:
            continue

        rotation_error, translation_error = errors
        # NOTE(brendan): Use the predictions from the bin with the most
        # matches.
        if len(matches_test) > max_matches:
            max_matches = len(matches_test)
            min_error.rotation = rotation_error
            min_error.translation = translation_error

    return min_error


def _get_screen_coords(view1, view2):
    """Return the screen coordinates of inlier points post-matching of the two
    views `view1` and `view2`.

    Also, flips the screen coordinates from (H, W) to (W, H), and subtracts the
    center such that (W/2, H/2) becomes (0, 0).
    """
    screen_coords = np.concatenate(
        [view1.inlier_points, view2.inlier_points], axis=1)
    screen_coords = screen_coords.reshape(
        [view1.inlier_points.shape[0], 2, 2])

    screen_coords -= np.array([240, 320])

    return np.flip(screen_coords, axis=-1)


def _vet_inliers(inliers, view1, view2, matches, ax):
    """Prompt the user to vet, i.e., to double-check, that all the inliers
    actually lie on the object.
    """
    for i, inlier in enumerate(inliers):
        print('Checking inlier')
        if not inlier:
            continue

        masked_inliers = np.zeros_like(inliers)
        masked_inliers[i] = inlier
        skimage.feature.plot_matches(ax=ax,
                                     image1=view1.img,
                                     image2=view2.img,
                                     keypoints1=view1.keypoints,
                                     keypoints2=view2.keypoints,
                                     matches=matches[masked_inliers],
                                     only_matches=True)

        plt.savefig('foo.png')
        plt.cla()

        click.echo('Keep?')
        char = click.getchar()
        if char.lower() == 'n':
            inliers[i] = False
        else:
            assert char.lower() == 'y'

    print(f'vetted inliers: {len([inlier for inlier in inliers if inlier])}')

    view1.inlier_points = view1.keypoints[matches[inliers, 0]]
    view2.inlier_points = view2.keypoints[matches[inliers, 1]]

    screen_coords = _get_screen_coords(view1, view2)

    return screen_coords, inliers, matches


def _solve(initial_guesses, loss_fn, bounds, args):
    """Run the solver to find the result to minimize `loss_fn`."""
    optimize_result = scipy.optimize.least_squares(
        loss_fn,
        initial_guesses,
        jac='2-point',
        bounds=bounds,
        method='trf',
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale=1.0,
        loss='soft_l1',
        f_scale=1.0,
        diff_step=None,
        tr_solver=None,
        tr_options={},
        jac_sparsity=None,
        max_nfev=100000,
        verbose=0,
        args=args,
        kwargs={})

    print(f'cost: {optimize_result.cost}')
    print(f'message: {optimize_result.message}')
    print(f'nfev: {optimize_result.nfev}')
    print(f'x: {optimize_result.x}')

    return optimize_result


@click.command()
@click.option('--objname',
              default='',
              help='Filename of YAML file for example.')
@click.option('--should-predict-pose/--no-should-predict-pose',
              default=False,
              help='If set, pose will be predicted from the matched '
                   'keypoints.')
def apc_read(objname, should_predict_pose):
    """Process raw APC dataset."""
    np.random.seed(0)

    descriptor_extractor = skimage.feature.ORB()

    if should_predict_pose:
        obj_basename = os.path.basename(objname)
        result_log = {}
        for shelf in ['A', 'D', 'G', 'J']:
            print(f'Evaluating shelf {shelf}!')

            view = _get_view(objname, descriptor_extractor, cam=1, shelf=shelf)
            error = _predict_pose(objname, descriptor_extractor, view)
            result_log[f'{obj_basename}/{shelf}/rotation_error'] = error.rotation
            result_log[f'{obj_basename}/{shelf}/translation_error'] = error.translation

        os.system('mkdir results')
        with open(f'results/{obj_basename}', 'w') as f:
            json.dump(result_log, f)

        exit()

    # NOTE(brendan): Depth coefficient computed from shelf F examples.
    depth_coef = 2.09e-5
    for shelf in ['B', 'C', 'E', 'F', 'H', 'I', 'K', 'L']:
        view1 = _get_view(objname, descriptor_extractor, cam=1, shelf=shelf)
        view2 = _get_view(objname, descriptor_extractor, cam=2, shelf=shelf)

        matches, inliers = _match_discard_outliers(view1, view2)

        print(f'Number of matches: {matches.shape[0]}')
        print(f'Number of inliers: {inliers.sum()}')

        _, ax = plt.subplots(nrows=1, ncols=1)
        plt.gray()

        screen_coords, inliers, matches = _vet_inliers(inliers,
                                                       view1,
                                                       view2,
                                                       matches,
                                                       ax)

        world_coords = []
        for i in range(screen_coords.shape[0]):
            initial_guess = np.zeros([2])
            args = (screen_coords[i:i + 1], [view1, view2], depth_coef)
            optimize_result = _solve(initial_guess,
                                     _triangulation_loss_depth,
                                     bounds=(-np.inf, np.inf),
                                     args=args)
            world_coords.append(optimize_result.x)

        directory = f'features/{objname}/{shelf}'
        os.system(f'mkdir -p {directory}')
        with open(f'{directory}/f.pkl', 'wb') as f:
            saved_matches = {
                'depth_coef': depth_coef,
                'inliers': inliers,
                'matches': matches,
                'screen_coords': screen_coords,
                'view1': view1,
                'view2': view2,
                'world_coords': world_coords,
            }
            pickle.dump(saved_matches, f)


if __name__ == '__main__':
    apc_read()  # pylint:disable=no-value-for-parameter
