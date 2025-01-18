import pycolmap
import numpy as np

# 读取 COLMAP 输出的 images.bin 文件
def load_images(images_bin_path):
    db = pycolmap.ImageDatabase(images_bin_path)
    return db.images()

# 将图像信息写入 txt 文件
def save_images_to_txt(images, output_txt_path):
    with open(output_txt_path, 'w') as f:
        for image_id, image in images.items():
            # 图像 ID
            f.write(f"Image ID: {image_id}\n")
            
            # 图像名称
            f.write(f"Image Name: {image.name}\n")
            
            # 旋转矩阵 (3x3)
            f.write("Rotation Matrix:\n")
            rotation_matrix = image.rotation
            for row in rotation_matrix:
                f.write(" ".join([f"{val:.6f}" for val in row]) + "\n")
            
            # 平移向量 (3,)
            f.write("Translation Vector:\n")
            translation_vector = image.translation
            f.write(" ".join([f"{val:.6f}" for val in translation_vector]) + "\n")
            
            # 匹配点信息
            f.write(f"Number of Points: {len(image.points)}\n")
            for point_id, point in image.points.items():
                f.write(f"  Point ID: {point_id} -> [Image Points: {point.image_point}]\n")
            
            f.write("\n" + "="*40 + "\n\n")

# 主函数
def main():
    images_bin_path = '/home/huangzitong/workspace/gaussian_splatter/PGSR/datasets/sampling/demo_removed/sparse/0/images.bin'  # 输入你的 images.bin 文件路径
    output_txt_path = '/home/huangzitong/workspace/gaussian_splatter/PGSR/datasets/sampling/demo_removed/sparse/0/images.txt'  # 输出 txt 文件路径
    
    # 加载 images.bin 文件
    images = load_images(images_bin_path)
    
    # 保存为 txt 文件
    save_images_to_txt(images, output_txt_path)
    print(f"Data has been written to {output_txt_path}")

if __name__ == "__main__":
    main()
