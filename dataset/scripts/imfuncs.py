import cv2
import numpy as np

class imfuncs():
    @staticmethod
    def random_rotation(image, only_90=False, only_180=False):
        if only_90:
            angle = np.random.choice([0, 90, 180, 270])
        elif only_180:
            angle = np.random.choice([0, 180])
        else:
            angle = np.random.uniform(-180, 180)
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return rotated

    @staticmethod
    def lowest_non_transparent_pixel(image):
        alpha_channel = image[:, :, 3]
        rows = image.shape[0]

        for row in range(rows - 1, -1, -1):
            if np.any(alpha_channel[row, :] > 0):
                return row
        return -1

    @staticmethod
    def paste_rgba(background, foreground, x, y):
        bg_slice = background[y:y+foreground.shape[0], x:x+foreground.shape[1]]
        h, w = bg_slice.shape[:2]
        fg_part = foreground[:h, :w]
        alpha_fg = fg_part[:, :, 3:4] / 255.0
        alpha_bg = bg_slice[:, :, 3:4] / 255.0

        out_rgb = alpha_fg * fg_part[:,:,:3] + (1 - alpha_fg) * bg_slice[:,:,:3]
        out_a   = alpha_fg + alpha_bg * (1 - alpha_fg)

        bg_slice[:, :, :3] = out_rgb.astype(np.uint8)
        bg_slice[:, :, 3] = (out_a[..., 0] * 255).astype(np.uint8)

    @staticmethod
    def get_shadow(image, intensity):
        shadow_color=(0,0,0,intensity)
        alpha_channel = image[:, :, 3]
        shadow = np.zeros_like(image, dtype=np.uint8)
        shadow[alpha_channel > 0] = shadow_color
        return shadow
    
    @staticmethod
    def resize_image(image, factor):
        new_size = (int(image.shape[1] * factor), int(image.shape[0] * factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image
    
    @staticmethod
    def manage_overlap(images):
        """
        given a list of equal size images with alpha channel, 
        reorder the list by non-transparent pixel count (ascending)
        and return the pasted ordered image
        """
        def count_non_transparent_pixels(img):
            return np.sum(img[:, :, 3] > 0)

        images_sorted = sorted(images, key=count_non_transparent_pixels, reverse=True)
        
        final_image = images_sorted[0].copy()
        for img in images_sorted[1:]:
            imfuncs.paste_rgba(final_image, img, 0, 0)

        return final_image
    
    @staticmethod
    def get_main_positions(horiz_bound, vert_bound, image):
        min_x = horiz_bound[0]
        max_x = max(min_x, horiz_bound[1] - image.shape[1])
        min_y = vert_bound[0]
        max_y = max(min_y, vert_bound[1] - image.shape[0])
        x_pos = np.random.randint(min_x, max_x + 1)
        y_pos = np.random.randint(min_y, max_y + 1)

        return x_pos, y_pos

    @staticmethod
    def get_random_edge_point(image_rgba):
        alpha = image_rgba[:, :, 3]
        mask = (alpha > 0).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        neighbor_count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        edge = (mask == 1) & (neighbor_count < 9)
        y_coords, x_coords = np.where(edge)
        idx = np.random.randint(0, len(y_coords))
        return (x_coords[idx], y_coords[idx])
    
    @staticmethod
    def get_secondary_positions(horiz_bound, vert_bound, canvas, image, shadow_offset):
        edg_x, edge_y = imfuncs.get_random_edge_point(canvas)
        sec_x_pos = edg_x - image.shape[1]//2
        sec_y_pos = edge_y - image.shape[0]//2
        sec_x_pos = np.clip(sec_x_pos, horiz_bound[0], horiz_bound[1] - image.shape[1])
        sec_y_pos = np.clip(sec_y_pos, vert_bound[0], vert_bound[1] - image.shape[0])
        
        return int(sec_x_pos), int(sec_y_pos)
    
    @staticmethod
    def get_bounding_box(image):
        alpha_channel = image[:, :, 3]
        ys, xs = np.where(alpha_channel > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        return (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, w, h)