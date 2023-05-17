import numpy as np
from src.lib import camera

# Definition of unit cube centered at the orign
x_width = 1.0
y_depth = 1.0
z_height = 1.0

_WORLD_T_POINTS = np.array([
    [0, 0, 0],  #0
    [0, 0, z_height],  #1
    [0, y_depth, z_height],  #2
    [0, y_depth, 0],  #3
    [x_width, 0, 0],  #4
    [x_width, 0, z_height],  #5
    [x_width, y_depth, z_height],  #6
    [x_width, y_depth, 0],  #7
]) - 0.5

def get_2d_bbox_of_9D_box(camera_T_object, scale_matrix, camera_model):
  unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
  morphed_homopoints = camera_T_object @ (scale_matrix @ unit_box_homopoints)
  morphed_pixels = camera.convert_homopixels_to_pixels(camera_model.K_matrix @ morphed_homopoints).T
  bbox = [
      np.array([np.min(morphed_pixels[:, 0]),
                np.min(morphed_pixels[:, 1])]),
      np.array([np.max(morphed_pixels[:, 0]),
                np.max(morphed_pixels[:, 1])])
  ]
  return bbox


def project_pose_onto_image(pose, camera_model):
  unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
  morphed_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  morphed_pixels = camera.convert_homopixels_to_pixels(camera_model.project(morphed_homopoints)).T
  morphed_pixels = morphed_pixels[:, ::-1]
  return morphed_pixels


def get_2d_bbox_of_projection(bbox_ext):
  bbox = [
      np.array([np.min(bbox_ext[:, 0]), np.min(bbox_ext[:, 1])]),
      np.array([np.max(bbox_ext[:, 0]), np.max(bbox_ext[:, 1])])
  ]
  return bbox


def define_control_points():
  return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


def compute_alphas(Xw, Cw):
  X = np.concatenate((Xw, np.array([np.ones((8))])), axis=0) # 4x8
  C = Cw.T # 4x3 --> 3x4
  C = np.concatenate((C, np.array([np.ones((4))])), axis=0) #4x4
  Alpha = np.matmul(np.linalg.inv(C), X)
  return Alpha.T


def construct_M_matrix(bbox_pixels, alphas, K_matrix):
  '''
  More detailed ePnP explanation: https://en.wikipedia.org/wiki/Perspective-n-Point
  '''
  M = np.zeros([16, 12]) # 16 is for bounding box verticies, 12 is for ? control pts?
  f_x = K_matrix[0, 0]
  f_y = K_matrix[1, 1]
  c_x = K_matrix[0, 2]
  c_y = K_matrix[1, 2]
  for ii in range(8):
    u = bbox_pixels[0, ii]
    v = bbox_pixels[1, ii]
    for jj in range(4):
      alpha = alphas[ii, jj]
      M[ii * 2, jj * 3] = f_x * alpha
      M[ii * 2, jj * 3 + 2] = (c_x - u) * alpha
      M[ii * 2 + 1, jj * 3 + 1] = f_y * alpha
      M[ii * 2 + 1, jj * 3 + 2] = (c_y - v) * alpha
  return M


def convert_control_to_box_vertices(control_points, alphas):
  bbox_vertices = np.zeros([8, 3])
  for i in range(8):
    for j in range(4):
      alpha = alphas[i, j]
      bbox_vertices[i] = bbox_vertices[i] + alpha * control_points[j]

  return bbox_vertices


def solve_for_control_points(M):
  e_vals, e_vecs = np.linalg.eig(M.T @ M)
  control_points = e_vecs[:, np.argmin(e_vals)]
  control_points = control_points.reshape([4, 3])
  return control_points


def compute_homopoints_from_control_points(camera_control_points, alphas, K_matrix):
  camera_points = convert_control_to_box_vertices(camera_control_points, alphas)
  camera_homopoints = camera.convert_points_to_homopoints(camera_points.T)
  return camera_homopoints
  unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)


def optimize_for_9D(bbox_pixels, camera_model, solve_for_transforms=False):
  K_matrix = camera_model.K_matrix
  Cw = define_control_points()
  Xw = _WORLD_T_POINTS # 8x3
  alphas = compute_alphas(Xw.T, Cw)
  M = construct_M_matrix(bbox_pixels, alphas, np.copy(K_matrix))
  camera_control_points = solve_for_control_points(M)
  camera_points = convert_control_to_box_vertices(camera_control_points, alphas)
  camera_homopoints = camera.convert_points_to_homopoints(camera_points.T)
  if solve_for_transforms:
    unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
    # Test both the negative and positive solutions of the control points and pick the best one. Taken from the Google MediaPipe Code base.
    error_one, camera_T_object_one, scale_matrix_one = estimateSimilarityUmeyama(
        unit_box_homopoints, camera_homopoints
    )

    camera_homopoints = compute_homopoints_from_control_points(
        -1 * camera_control_points, alphas, K_matrix
    )
    error_two, camera_T_object_two, scale_matrix_two = estimateSimilarityUmeyama(
        unit_box_homopoints, camera_homopoints
    )
    if error_one < error_two:
      camera_T_object = camera_T_object_one
      scale_matrix = scale_matrix_one
    else:
      camera_T_object = camera_T_object_two
      scale_matrix = scale_matrix_two

    # Compute Fit to original pixles:
    morphed_points = camera_T_object @ (scale_matrix @ unit_box_homopoints)
    morphed_pixels = points_to_camera(morphed_points, K_matrix)
    confidence = np.linalg.norm(bbox_pixels - morphed_pixels)
    return confidence, camera_T_object, scale_matrix
  camera_homopixels = K_matrix @ camera_homopoints
  return camera.convert_homopixels_to_pixels(camera_homopixels).T


def estimateSimilarityUmeyama(source_hom, TargetHom):
  # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
  assert source_hom.shape[0] == 4
  assert TargetHom.shape[0] == 4
  SourceCentroid = np.mean(source_hom[:3, :], axis=1)
  TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
  nPoints = source_hom.shape[1]

  CenteredSource = source_hom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
  CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

  CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

  if np.isnan(CovMatrix).any():
    print('nPoints:', nPoints)
    print('source_hom', source_hom.shape)
    print('TargetHom', TargetHom.shape)
    raise RuntimeError('There are NANs in the input.')

  U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
  d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
  if d:
    D[-1] = -D[-1]
    U[:, -1] = -U[:, -1]

  Rotation = np.matmul(U, Vh)
  var_source = np.std(CenteredSource[:3, :], axis=1)
  var_target_aligned = np.std(np.linalg.inv(Rotation) @ CenteredTarget[:3, :], axis=1)
  ScaleMatrix = np.diag(var_target_aligned / var_source)

  Translation = TargetHom[:3, :].mean(axis=1) - source_hom[:3, :].mean(axis=1).dot(
      ScaleMatrix @ Rotation.T
  )

  source_T_target = np.identity(4)
  source_T_target[:3, :3] = Rotation
  source_T_target[:3, 3] = Translation
  scale_matrix = np.eye(4)
  scale_matrix[0:3, 0:3] = ScaleMatrix
  # Measure error fit
  morphed_points = source_T_target @ (scale_matrix @ source_hom)
  error = np.linalg.norm(morphed_points - TargetHom)
  return error, source_T_target, scale_matrix


def points_to_camera(world_T_homopoints, K_matrix):
  homopixels = K_matrix @ world_T_homopoints
  return camera.convert_homopixels_to_pixels(homopixels)


def find_absolute_scale(new_z, camera_T_object, object_scale, debug=True):
  old_z = camera_T_object[2, 3]
  abs_camera_T_object = np.copy(camera_T_object)
  abs_camera_T_object[0:3, 3] = (new_z / old_z) * abs_camera_T_object[0:3, 3]
  abs_object_scale = np.eye(4)
  abs_object_scale[0:3, 0:3] = (new_z / old_z) * object_scale[0:3, 0:3]
  return abs_camera_T_object, abs_object_scale
