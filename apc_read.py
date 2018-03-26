"""Amazon Picking Challenge dataset reader."""
import numpy as np
import os
import yaml

import click
import matplotlib.pyplot as plt
import scipy.optimize
import skimage.data
import skimage.color
import skimage.feature
import skimage.measure
import skimage.transform
import torch


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


def _triangulation_loss(world_coords, screen_coords, poses):
    """Minimize the residuals by directly reprojecting the estimated world
    coordinates of a point, and comparing against the known screen coordinates
    of that point.

    Args:
        world_coords: A matrix of shape (N, 3) of N points (x_c, y_c, z_c),
            each of which is an estimate of the world coordinates for the given
            point. Units: meters.
        screen_coords: A matrix of shape (N, M, 2), where N is the number of data
            points corresponding to the given keypoint the world coordinates of
            which are being estimated, M is the number of poses, and there are
            two coordinates (x_s, y_s) for each pose and data point
            combination. Units: pixels.
        poses: A list of length M of the pose labels from the APC dataset. Pose
            rotations are unitless values (?), and translations are in meters.

    Returns:
        The MSE between x_s and the re-projected x_s'.
    """
    # NOTE(brendan): The focal length is 525px. Source:
    # http://wiki.ros.org/kinect_calibration/technical#Lens_distortion_and_focal_length
    focal_length = 525
    world_coords = world_coords.reshape([screen_coords.shape[0], 3])

    mse = 0.0
    for i, pose in enumerate(poses):
        rotation = pose['object_rotation_wrt_camera']['data']
        translation = pose['object_translation_wrt_camera']

        r0 = rotation[:1*3]
        r1 = rotation[1*3:2*3]
        r2 = rotation[2*3:]

        t_x = translation[0]
        t_y = translation[1]
        t_z = translation[2]

        # NOTE(brendan): The values to estimate.
        x_c = world_coords[:, 0]
        y_c = world_coords[:, 1]
        z_c = world_coords[:, 2]

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
        mse += np.matmul(error, error.transpose()).mean()

    mse /= len(poses)

    return mse


@click.command()
@click.option('--objname',
              default='',
              help='Filename of YAML file for example.')
def apc_read(objname):
    """Process raw APC dataset."""
    np.random.seed(0)

    pose_view1 = _get_pose(f'{objname}-pose-F-1-1-0.yml')
    pose_view2 = _get_pose(f'{objname}-pose-F-1-2-0.yml')
    img_view1 = skimage.io.imread(f'{objname}-image-F-1-1-0.png')
    img_view2 = skimage.io.imread(f'{objname}-image-F-1-2-0.png')

    img_view1, img_view2 = map(skimage.color.rgb2gray, (img_view1, img_view2))

    descriptor_extractor = skimage.feature.ORB()

    keypoints_left, descriptors_left = _detect_and_extract(
        img_view1, descriptor_extractor)
    keypoints_right, descriptors_right = _detect_and_extract(
        img_view2, descriptor_extractor)

    matches = skimage.feature.match_descriptors(descriptors_left,
                                                descriptors_right,
                                                cross_check=True)

    data = (keypoints_left[matches[:, 0]], keypoints_right[matches[:, 1]])
    model, inliers = skimage.measure.ransac(
        data,
        model_class=skimage.transform.FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=1,
        max_trials=5000)

    inliers_keypoints_left = keypoints_left[matches[inliers, 0]]
    inliers_keypoints_right = keypoints_right[matches[inliers, 1]]

    num_inliers = inliers.sum()
    print(f'Number of matches: {matches.shape[0]}')
    print(f'Number of inliers: {inliers.sum()}')

    screen_coords = np.concatenate(
        [inliers_keypoints_left, inliers_keypoints_right], axis=1)
    screen_coords = screen_coords.reshape([num_inliers, 2, 2])

    screen_coords -= np.array([240, 320])
    screen_coords = np.flip(screen_coords, axis=-1)

    poses = [pose_view1, pose_view2]

    initial_guesses = np.array([0.0, 0.0, 1.0])
    initial_guesses = initial_guesses.reshape([1, 3])
    initial_guesses = initial_guesses.repeat(num_inliers, axis=0)
    initial_guesses = initial_guesses.reshape([3*num_inliers])

    # NOTE(brendan): Take the first N points
    num_points = 10
    start_point = 20
    end_point = start_point + num_points
    for i in range(start_point, end_point):
        optimize_result = scipy.optimize.least_squares(_triangulation_loss,
                                                       initial_guesses[3*i:3*(i + 1)],
                                                       jac='2-point',
                                                       bounds=(-np.inf, np.inf),
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
                                                       max_nfev=10000,
                                                       verbose=0,
                                                       args=(screen_coords[i:i + 1], poses),
                                                       kwargs={})

        print(f'cost: {optimize_result.cost}')
        print(f'message: {optimize_result.message}')
        print(f'nfev: {optimize_result.nfev}')
        print(f'x: {optimize_result.x}')

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()

    skimage.feature.plot_matches(ax=ax,
                                 image1=img_view1,
                                 image2=img_view2,
                                 keypoints1=keypoints_left,
                                 keypoints2=keypoints_right,
                                 matches=matches[inliers],
                                 only_matches=True)
    plt.show()


if __name__ == '__main__':
    apc_read()  # pylint:disable=no-value-for-parameter
