import numpy as np


def find_covariance(scene_pd: "np.array['3,N', float]") -> "np.array['3,3', float]":
    scene_mean = scene_pd.mean(axis=-1).reshape((3,1))
    scene_hat = scene_pd - scene_mean
    scene_hat = scene_hat[:, np.random.choice(np.arange(scene_hat.shape[1]), size=min(1000, scene_hat.shape[1]//10), replace=False)]
    covariance = np.cov(scene_hat)
    return covariance


def covariance_obj(obj:"open3d.geometry.TriangleMesh", rotation:"np.array['3,3', float]") -> "np.array['3,3', float]":
    import open3d
    pcd = open3d.geometry.TriangleMesh.sample_points_uniformly(obj, number_of_points=10000)
    points = np.asarray(pcd.points).T
    points_t = rotation @ points
    C = find_covariance(points_t)
    return C


def covariance_mesh(obj: "bpy.types.Object", rotation:"np.array['3,3', float]") -> "np.array['3,3', float]":
    import bpy
    coords = np.array([(obj.matrix_world @ v.co) for v in obj.data.vertices]) # Nx3
    coords = coords.T # 3xN
    points_t = rotation @ coords# 3xN
    C = find_covariance(points_t)
    return C


def covariance2rotation(covariance: "np.array['3,3', float]") -> "np.array['3,3', float]":
    U, s, V = np.linalg.svd(covariance, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(V)) < 0.0
    if d:
        s[-1] = -s[-1]
        U[:, -1] = -U[:, -1]
    return U


def rand_rotation_matrix(deflection=1.0):
    theta, phi, z = np.random.uniform(size=(3,))
    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.
    r = np.sqrt(z)
    V = (np.sin(phi) * r,
         np.cos(phi) * r,
         np.sqrt(2.0 - z))
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def test_r_svd():
    import open3d
    r = rand_rotation_matrix()
    obj = open3d.io.read_triangle_mesh("../KeyPose/Model&Keypoint/mug_0.obj")
    pcd = open3d.geometry.TriangleMesh.sample_points_uniformly(obj, number_of_points=10000)
    points = np.asarray(pcd.points).T
    points_t = r @ points + np.random.rand(3, 1)
    C = find_covariance(points_t)
    r_hat = covariance2rotation(C)
    print(r - r_hat)


if __name__ == "__main__":
    test_r_svd()
