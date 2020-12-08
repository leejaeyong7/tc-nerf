import cv2
def to_depth_image(depth):
    '''
    Given HxWx1 depth image, returns nice visualization of the depth image
    '''
    d = depth.detach().cpu().numpy()
    if (d > 0).sum() == 0:
        depth_image = d.astype('uint8')
        depth_colored_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_PARULA)
        depth_colored_image[(d == 0).repeat(3, 2)] = 0

        return depth_colored_image[:, :, ::-1].transpose(2, 0, 1)

    min_d = d[d > 0].min()
    max_d = d.max()
    depth_image = (((max_d - d) / (max_d - min_d)) * 255).astype('uint8')
    depth_colored_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_PARULA)
    depth_colored_image[(d == 0).repeat(3, 2)] = 0

    return depth_colored_image[:, :, ::-1].transpose(2, 0, 1)