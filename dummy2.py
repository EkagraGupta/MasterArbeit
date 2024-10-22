import os

def display_folder_overview(root_dir):
    """Display the folder structure with image counts and first three image filenames."""
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            # Count the number of JPEG files in the class directory
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.JPEG')]
            image_count = len(image_files)

            # Display the class directory name and the count of images
            print(f"{class_name}: {image_count} images")

            # Display the first three image filenames if they exist
            for i, image_name in enumerate(image_files[:3]):
                print(f"    |-- {image_name}")

if __name__ == '__main__':
    root_directory = '/home/ekagra/Documents/GitHub/MasterArbeit/data/tiny_imnet/tiny-imagenet-200'  # Change this to your target directory
    display_folder_overview(root_directory)
