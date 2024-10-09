import numpy as np
import cv2  
import matplotlib.pyplot as plt


image = cv2.imread('data\\fakeflower.png')  
mean = 0
sigma = 25  # 噪声标准差，可以调整
#gaussian_noise = np.random.normal(mean, sigma, image.shape)
# 生成高斯噪声
gaussian_noise = np.random.normal(mean, sigma, image.shape)+80

# 确保噪声值在0到255之间
gaussian_noise = np.clip(gaussian_noise, 0, 255).astype(np.uint8)
print(image.shape)
# 显示噪声图像
#cv2.imshow('Gaussian Noise', gaussian_noise)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 保存噪声图像
#cv2.imwrite('data\\gaussian_noise.jpg', gaussian_noise)


# 绘制直方图
plt.hist(gaussian_noise.flatten(), bins=100, density=True)
plt.title('Histogram of Gaussian Noise')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid()
plt.show()

