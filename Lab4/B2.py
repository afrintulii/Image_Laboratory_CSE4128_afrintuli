import cv2
import numpy as np


def notch(M, N, points):
    H = np.ones((M, N), np.float32)
    for u, v in points:

        for i in range(u - 1, u + 2):
            for j in range(v - 1, v + 2):
                if 0 <= i < M and 0 <= j < N:
                    H[i, j] = 0.0
                    H[(M - 1) - i, (N - 1) - j] = 0.0
    return H


img = cv2.imread('two_noise.jpeg', 0)



ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

points = [(300, 300), (262, 261)]

notch_filter = notch(img.shape[0], img.shape[1], points)

result = magnitude_spectrum * notch_filter
result = 20 * np.log(np.abs(result)+1)
result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

final_result = np.multiply(result, np.exp(1j*ang))


img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("input ",img)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
cv2.imshow("Angle", ang_)
cv2.imshow("notch filter", notch_filter)
cv2.imshow("multiplied", result_normalized)
# cv2.imshow("Magnitude Spectrum normalized", magnitude_spectrum_normalized)
cv2.imshow("Inverse transform", img_back_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()