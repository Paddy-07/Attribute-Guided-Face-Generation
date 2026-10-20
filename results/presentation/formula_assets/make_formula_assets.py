from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(r'E:\Me\Sem-5\Dl\Face_Gen\outputs\formula_assets')
OUT.mkdir(parents=True, exist_ok=True)
BG = (218, 215, 205)  # sampled from the user's supplied swatch
BLACK = (0, 0, 0)
FONT_PATH = r'C:\Windows\Fonts\segoeui.ttf'
W, H = 1920, 1080

slides = {
    '01_standard_gan': [
        'min_G max_D V(D, G) = Eₓ[log D(x)] + E_z[log(1 − D(G(z)))]',
    ],
    '02_conditional_generator': [
        'x_fake = G(z, a)',
    ],
    '03_training_conditions': [
        '(x_real, y_real)',
        'z ~ N(0, I₁₀₀)',
        'aᵢ ~ Bernoulli(0.5),    i = 1, …, 10',
        'x_fake = G(z, a)',
    ],
    '04_discriminator_outputs': [
        'D(x) = (r, q)',
        'r is real-valued,    q is a 10-dimensional vector',
    ],
    '05_preprocessing': [
        'x_norm = (x − 0.5) / 0.5 = 2x − 1',
        'y = (y_raw + 1) / 2,    y is in {0, 1}',
    ],
    '06_project_losses': [
        'BCEWithLogits(s, t) = −[t log σ(s) + (1 − t) log(1 − σ(s))]',
        'L_D = BCE(r_real, 0.9) + BCE(r_fake, 0) + λ · BCE(q_real, y_real)',
        'L_G = BCE(r_fake, 1) + λ · BCE(q_fake, a)',
        'λ = 1.0',
    ],
}

for name, lines in slides.items():
    im = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(im)
    # A generous line size makes these legible when placed into a slide.
    font_size = 58 if len(lines) <= 2 else 50
    font = ImageFont.truetype(FONT_PATH, font_size)
    gap = 38
    heights = [draw.textbbox((0, 0), s, font=font)[3] for s in lines]
    total = sum(heights) + gap * (len(lines) - 1)
    y = (H - total) // 2
    for s, h in zip(lines, heights):
        draw.text((100, y), s, font=font, fill=BLACK)
        y += h + gap
    im.save(OUT / f'{name}.png', dpi=(144, 144))

# Contact sheet for a quick preview of every formula asset.
thumb_w, thumb_h = 960, 540
sheet = Image.new('RGB', (thumb_w * 2, thumb_h * 3), BG)
for i, name in enumerate(slides):
    img = Image.open(OUT / f'{name}.png').resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    sheet.paste(img, ((i % 2) * thumb_w, (i // 2) * thumb_h))
sheet.save(OUT / 'all_formulas_preview.png', dpi=(144, 144))
print('\n'.join(str(p) for p in sorted(OUT.glob('*.png'))))

