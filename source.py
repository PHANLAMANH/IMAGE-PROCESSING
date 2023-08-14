from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Function to change brightness
def change_brightness(image, value):
    image = image.astype("int16")
    image = image + value
    image = np.clip(image, 0, 255)
    return image.astype("uint8")


# Function to change contrast
def change_contrast(image, contrast):
    # Calculate the contrast factor
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))

    # Apply the contrast adjustment using a Look Up Table (LUT)
    lut = np.arange(256) * factor
    lut = np.clip(lut, 0, 255).astype("uint8")
    contrast_image = lut[image]

    return contrast_image


# Function to flip image (horizontal or vertical)
def flip_image(image, direction):
    if direction == "horizontal":
        return np.flip(image, axis=1)
    elif direction == "vertical":
        return np.flip(image, axis=0)
    else:
        return image


# Function to convert RGB image to grayscale
def rgb_to_grayscale(image):
    grayscale_image = np.mean(image, axis=2).astype(np.uint8)

    return grayscale_image


# Function to convert RGB image to sepia tone
def rgb_to_sepia(img_2d: np.ndarray) -> np.ndarray:
    h, w, c = img_2d.shape
    new_img_1d = img_2d[:, :, :3].reshape((h * w, 3)).astype(float)
    formula = [[0.393, 0.349, 0.272], [0.769, 0.686, 0.534], [0.189, 0.168, 0.131]]
    new_img_1d = np.clip(np.matmul(new_img_1d, formula), 0, 255)

    return new_img_1d.reshape((h, w, 3)).round(0).astype(np.uint8)


def convert_to_rgb(image):
    # Convert RGBA image to RGB
    rgb_image = image.convert("RGB")

    return rgb_image


# Function to apply blur or sharpen filters using convolution


def kernel_sharpening(image, kernel):
    # Apply convolution to each channel of the image
    sharpened_image = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[2]):
        sharpened_image[:, :, channel] = convolve(image[:, :, channel], kernel)

    # Clip values to ensure they are within the valid range
    sharpened_image = np.clip(sharpened_image, 0, 255)

    return sharpened_image.astype(np.uint8)


def sharpening_image(channel):
    kernel_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen_image_channel = np.zeros_like(channel)

    # Pad the channel with zeros to handle edges
    padded_channel = np.pad(
        channel, ((1, 1), (1, 1)), mode="constant", constant_values=0
    )

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            sharpen_image_channel[i, j] = np.clip(
                np.sum(padded_channel[i : i + 3, j : j + 3] * kernel_sharpening),
                0,
                255,
            )
    return sharpen_image_channel


def rgb_to_sharpen(image, factor):
    image_array = np.array(image)
    # separate the channels
    r = image_array[:, :, 0]
    g = image_array[:, :, 1]
    b = image_array[:, :, 2]
    # sharpen each channel
    sharpen_r = sharpening_image(r)
    sharpen_g = sharpening_image(g)
    sharpen_b = sharpening_image(b)
    # combine the channels
    sharpen_image_array = np.dstack((sharpen_r, sharpen_g, sharpen_b))
    sharpen_image_array = sharpen_image_array.astype(np.uint8)
    sharpen_image = Image.fromarray(sharpen_image_array)
    return sharpen_image  # Return the Image object


# Convolution function
def convolve(channel, kernel):
    k_rows, k_cols = kernel.shape
    padded_channel = np.pad(
        channel, ((k_rows // 2, k_rows // 2), (k_cols // 2, k_cols // 2)), mode="edge"
    )
    result = np.zeros_like(channel, dtype=np.float32)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            result[i, j] = np.sum(
                padded_channel[i : i + k_rows, j : j + k_cols] * kernel
            )

    return result


def rgb_to_blur(image, kernel_size):
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Check the number of color channels in the image
    num_channels = image_array.shape[2] if len(image_array.shape) == 3 else 1

    # Define the blurring kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    # Apply the blurring kernel to each color channel (or the grayscale channel)
    blurred_image_array = np.zeros_like(image_array, dtype=np.float32)

    for channel in range(num_channels):
        blurred_image_array[..., channel] = convolve(image_array[..., channel], kernel)

    # Clip values to ensure they are within the valid range
    blurred_image_array = np.clip(blurred_image_array, 0, 255)

    # Convert the blurred image array to an 8-bit integer array
    blurred_image_array = blurred_image_array.astype(np.uint8)

    # Convert the numpy array to an Image object
    blurred_image = Image.fromarray(blurred_image_array)

    return blurred_image  # Return the Image object


# Function to crop image to specified size
def crop_image(image_path, crop_height, crop_width, shape):
    # Open the image and convert it into a numpy array
    img = Image.open(image_path)
    img_np = np.array(img)

    # Calculate the center of the image and the coordinates of the rectangle you want to crop
    h, w, _ = img_np.shape
    ch, cw = crop_height, crop_width
    x1, y1 = w // 2 - cw // 2, h // 2 - ch // 2
    x2, y2 = w // 2 + cw // 2, h // 2 + ch // 2

    # Crop the image
    cropped_np = img_np[y1:y2, x1:x2]

    # Create a mask based on the shape
    center_y, center_x = cropped_np.shape[0] // 2, cropped_np.shape[1] // 2
    Y, X = np.ogrid[:ch, :cw]
    if shape == "circle":
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        radius = min(center_x, center_y)
        mask = dist_from_center <= radius
    elif shape == "rectangle":
        mask = np.ones((ch, cw), dtype=bool)
    elif shape == "ellipse":
        rx, ry = cw / 2, ch / 2
        mask1 = ((X - center_x) / rx) ** 2 + ((Y - center_y) / ry) ** 2 <= 1
        mask2 = ((X - center_x) / ry) ** 2 + ((Y - center_y) / rx) ** 2 <= 1
        mask = np.logical_or(mask1, mask2)
    else:
        raise ValueError("Invalid shape")

    num_channels = cropped_np.shape[2]
    mask = np.stack([mask] * num_channels, axis=-1)

    # Apply the mask to the image
    masked_img_np = np.where(mask, cropped_np, 0)  # Use 0 for black

    # Convert the masked image back to a PIL image
    masked_img = Image.fromarray(masked_img_np.astype(np.uint8))

    return masked_img


def process_image(image_name, function_id):
    # Read the image
    image = Image.open(image_name)
    # Convert image to numpy array
    image_array = np.array(image)

    if function_id == 1:  # Change Brightness
        brightness = int(input("Enter the brightness value (-255 to 255): "))
        processed_image = change_brightness(image_array, brightness)

        # Convert the numpy array back to an Image object
        processed_image = Image.fromarray(processed_image)

        # Save the processed image
        processed_image_name = f"brightness_{brightness}.png"
        processed_image.save(processed_image_name)
        print("Processed image saved as", processed_image_name)

    elif function_id == 2:  # Change Contrast
        contrast = int(input("Enter the contrast value (-255 to 255): "))
        processed_image = change_contrast(image_array, contrast)

        # Convert the numpy array back to an Image object
        processed_image = Image.fromarray(processed_image)

        # Save the processed image
        processed_image_name = f"contrast_{contrast}.png"
        processed_image.save(processed_image_name)
        print("Processed image saved as", processed_image_name)

    elif function_id == 3:  # Flip Image
        direction = input("Enter the direction to flip (horizontal/vertical): ")
        processed_image = flip_image(image_array, direction)

        # Convert the numpy array back to an Image object
        processed_image = Image.fromarray(processed_image)

        # Save the processed image
        processed_image_name = f"flip_{direction}.png"
        processed_image.save(processed_image_name)
        print("Processed image saved as", processed_image_name)

    elif function_id == 4:  # Convert to Grayscale/Sepia Tone
        conversion_type = input("Enter the conversion type (grayscale/sepia): ")
        if conversion_type == "grayscale":
            processed_image = rgb_to_grayscale(image_array)
            processed_image_name = "grayscale.png"
        elif conversion_type == "sepia":
            processed_image = rgb_to_sepia(image_array)
            processed_image_name = "sepia.png"
        else:
            print("Invalid conversion type.")
            return

        # Convert the numpy array back to an Image object
        processed_image = Image.fromarray(processed_image)

        # Save the processed image
        processed_image.save(processed_image_name)
        print("Processed image saved as", processed_image_name)

    elif function_id == 5:  # Apply Blur Sharpen
        filter_type = input("Enter the filter type (blur/sharpen): ")
        if filter_type == "blur":
            radius = int(input("Enter the factor of the blur kernel: "))
            processed_image = rgb_to_blur(image_array, radius)
        elif filter_type == "sharpen":
            factor = int(input("Enter the factor of the sharpen kernel: "))
            processed_image = rgb_to_sharpen(
                image_array, factor
            )  # Use the Image object here

        else:
            print("Invalid filter type.")
            return

        # Save the processed image
        processed_image_name = f"{image_name}_processed.png"
        processed_image.save(processed_image_name)
        print("Processed image saved as", processed_image_name)

    elif function_id == 6:  # Crop Image
        width = int(input("Enter the width of the cropped image: "))
        height = int(input("Enter the height of the cropped image: "))
        shape = input(
            "Enter the shape of the cropped image (rectangle/circle/ellipse): "
        )
        processed_image = crop_image(image_name, height, width, shape)
        processed_image = crop_image(image_name, height, width, shape)
        if not isinstance(processed_image, np.ndarray):
            processed_image = np.array(processed_image)
        if processed_image.dtype != np.uint8:
            processed_image = processed_image.astype(np.uint8)
            processed_image = Image.fromarray(processed_image)
        # convert to pillow image
        processed_image = Image.fromarray(processed_image)
        # Save the processed image
        processed_image_name = f"crop_{width}x{height}_{shape}.png"
        processed_image.save(processed_image_name)
        print("Processed image saved as", processed_image_name)

    else:
        print("Invalid function ID.")
        return

    # Display the ỏiginal image and the processed image side by side using matplotlib subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Original Image")
    ax2.imshow(processed_image, cmap="gray")
    ax2.axis("off")
    ax2.set_title("Processed Image")
    plt.show()

    # Save the processed image
    processed_image.save("processed_image.png")


# Main function
def main():
    # Get the image file name from the user
    image_name = input("Enter the image file name: ")

    # Display the menu
    print("Image Processing Menu:")
    print("1. Change Brightness")
    print("2. Change Contrast")
    print("3. Flip Image")
    print("4. Convert to Grayscale/Sepia Tone")
    print("5. Apply Blur/Sharpen Filter")
    print("6. Crop Image(including shape)")
    print("0. Doesnt have execute all functions")

    # Get the function ID from the user
    function_id = int(input("Enter the function ID: "))

    # Process the image
    process_image(image_name, function_id)


# Run the main function
if __name__ == "__main__":
    main()
