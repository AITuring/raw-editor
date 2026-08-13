# RAW Editor — macOS Liquid Glass source

The files in `layers/` are intentionally flatter than the rendered preview. Icon Composer should own the platform mask, refraction, blur, shadow, and specular response.

## Import order

1. Canvas: use `#E7DDD0` for Default and `#171A1C` for Dark. The two `00-background-*.svg` files are reference swatches/fallback layers.
2. Group 1: `10-camera.svg` — enable Liquid Glass; use medium translucency and low-to-medium frost. In Dark appearance, raise the camera fill toward `#4A5053`.
3. Group 2: `20-aperture.svg` — keep Liquid Glass disabled so the vector aperture colors and blade boundaries stay crisp at every size.

## Appearance mapping

- Default: warm stone canvas, graphite camera, original aperture.
- Dark: charcoal canvas, lighter graphite camera, original aperture.
- Mono / tinted: map the camera to the dominant tint and the aperture to neutral tonal steps; preserve the transparent center.
- Clear Light / Clear Dark: keep the camera group translucent, but leave the aperture group more opaque than the camera for recognition.

The production canvas is 1024 × 1024. Do not add a rounded-rectangle mask to imported artwork; macOS applies it automatically.
