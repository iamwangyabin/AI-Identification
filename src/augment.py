from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class PostProcessConfig:
    probability: float = 0.5
    max_ops: int = 2
    jpeg_quality_min: int = 35
    jpeg_quality_max: int = 95
    webp_quality_min: int = 35
    webp_quality_max: int = 95
    blur_radius_max: float = 1.6
    resize_scale_min: float = 0.5
    noise_std_max: float = 4.0
    crop_scale_min: float = 0.75
    sharpen_factor_max: float = 2.0
    brightness_delta: float = 0.2
    contrast_delta: float = 0.2
    saturation_delta: float = 0.2
    gamma_delta: float = 0.2


class RandomPostProcessPerturbation:
    """Apply common post-processing artifacts to improve robustness."""

    def __init__(self, config: PostProcessConfig | None = None) -> None:
        self.config = config or PostProcessConfig()

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.config.probability:
            return image

        ops = [
            self._jpeg_compress,
            self._webp_compress,
            self._gaussian_blur,
            self._median_blur,
            self._crop_rescale,
            self._rescale_restore,
            self._resample_restore,
            self._sharpen,
            self._brightness_contrast_saturation,
            self._gamma_adjust,
            self._gaussian_noise,
        ]
        num_ops = random.randint(1, max(1, self.config.max_ops))
        for operation in random.sample(ops, k=min(num_ops, len(ops))):
            image = operation(image)
        return image

    def _jpeg_compress(self, image: Image.Image) -> Image.Image:
        quality = random.randint(
            self.config.jpeg_quality_min,
            self.config.jpeg_quality_max,
        )
        return self._compress(image, format_name="JPEG", quality=quality)

    def _webp_compress(self, image: Image.Image) -> Image.Image:
        quality = random.randint(
            self.config.webp_quality_min,
            self.config.webp_quality_max,
        )
        try:
            return self._compress(image, format_name="WEBP", quality=quality)
        except OSError:
            return self._jpeg_compress(image)

    def _compress(self, image: Image.Image, format_name: str, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format=format_name, quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _gaussian_blur(self, image: Image.Image) -> Image.Image:
        radius = random.uniform(0.1, self.config.blur_radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _median_blur(self, image: Image.Image) -> Image.Image:
        kernel_size = random.choice((3, 5))
        return image.filter(ImageFilter.MedianFilter(size=kernel_size))

    def _rescale_restore(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(self.config.resize_scale_min, 0.95)
        resized_size = (
            max(16, int(width * scale)),
            max(16, int(height * scale)),
        )
        resized = image.resize(resized_size, resample=self._random_resample())
        return resized.resize((width, height), resample=self._random_resample())

    def _crop_rescale(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_scale = random.uniform(self.config.crop_scale_min, 0.98)
        crop_w = max(16, int(width * crop_scale))
        crop_h = max(16, int(height * crop_scale))
        if crop_w >= width or crop_h >= height:
            return image
        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((width, height), resample=self._random_resample())

    def _resample_restore(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(max(self.config.resize_scale_min, 0.6), 0.95)
        resized_size = (
            max(16, int(width * scale)),
            max(16, int(height * scale)),
        )
        down = image.resize(resized_size, resample=self._random_resample())
        return down.resize((width, height), resample=self._random_resample())

    def _sharpen(self, image: Image.Image) -> Image.Image:
        factor = random.uniform(1.05, self.config.sharpen_factor_max)
        return ImageEnhance.Sharpness(image).enhance(factor)

    def _brightness_contrast_saturation(self, image: Image.Image) -> Image.Image:
        brightness = random.uniform(
            1.0 - self.config.brightness_delta,
            1.0 + self.config.brightness_delta,
        )
        contrast = random.uniform(
            1.0 - self.config.contrast_delta,
            1.0 + self.config.contrast_delta,
        )
        saturation = random.uniform(
            1.0 - self.config.saturation_delta,
            1.0 + self.config.saturation_delta,
        )
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def _gamma_adjust(self, image: Image.Image) -> Image.Image:
        gamma = random.uniform(
            1.0 - self.config.gamma_delta,
            1.0 + self.config.gamma_delta,
        )
        gamma = max(gamma, 0.05)
        lut = [min(255, max(0, round(255.0 * ((i / 255.0) ** gamma)))) for i in range(256)]
        return image.point(lut * 3)

    def _gaussian_noise(self, image: Image.Image) -> Image.Image:
        std = random.uniform(0.5, self.config.noise_std_max)
        array = np.asarray(image).astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=std, size=array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    @staticmethod
    def _random_resample() -> Image.Resampling:
        return random.choice(
            [
                Image.Resampling.NEAREST,
                Image.Resampling.BILINEAR,
                Image.Resampling.BICUBIC,
                Image.Resampling.LANCZOS,
            ]
        )
