from pathlib import Path
import sys
sys.path.insert(0, r'E:\Me\Sem-5\Dl\Face_Gen\outputs\formula_assets\python_deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT=Path(r'E:\Me\Sem-5\Dl\Face_Gen\outputs\formula_assets')
BG='#DAD7CD'
BLACK='#000000'
# Each entry is a separate, mathematically typeset equation image.
items=[
('eq_01_standard_gan', r'$\min_G\max_D\ V(D,G)=\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$'),
('eq_02_conditional_generator', r'$x_{\mathrm{fake}}=G(z,a)$'),
('eq_03_noise_distribution', r'$z\sim\mathcal{N}(0,I_{100})$'),
('eq_04_attribute_sampling', r'$a_i\sim\mathrm{Bernoulli}(0.5),\qquad i=1,\ldots,10$'),
('eq_05_discriminator_outputs', r'$D(x)=(r,q),\qquad r\in\mathbb{R},\quad q\in\mathbb{R}^{10}$'),
('eq_06_normalization', r'$x_{\mathrm{norm}}=\frac{x-0.5}{0.5}=2x-1$'),
('eq_07_label_conversion', r'$y=\frac{y_{\mathrm{raw}}+1}{2}\in\{0,1\}$'),
('eq_08_bce_with_logits', r'$\mathrm{BCEWithLogits}(s,t)=-\left[t\log\sigma(s)+(1-t)\log\left(1-\sigma(s)\right)\right]$'),
('eq_09_discriminator_loss', r'$L_D=\mathrm{BCE}(r_{\mathrm{real}},0.9)+\mathrm{BCE}(r_{\mathrm{fake}},0)+\lambda\,\mathrm{BCE}(q_{\mathrm{real}},y_{\mathrm{real}})$'),
('eq_10_generator_loss', r'$L_G=\mathrm{BCE}(r_{\mathrm{fake}},1)+\lambda\,\mathrm{BCE}(q_{\mathrm{fake}},a)$'),
('eq_11_lambda', r'$\lambda=1.0$'),
]
for name, expr in items:
    fig=plt.figure(figsize=(16,2.4), dpi=150, facecolor=BG)
    ax=fig.add_axes([0,0,1,1], facecolor=BG)
    ax.set_axis_off()
    fontsize=22 if name in ('eq_01_standard_gan','eq_08_bce_with_logits','eq_09_discriminator_loss') else 36
    ax.text(0.045,0.5,expr,color=BLACK,fontsize=fontsize,ha='left',va='center')
    fig.savefig(OUT/f'{name}.png',dpi=150,facecolor=BG)
    fig.savefig(OUT/f'{name}.svg',facecolor=BG)
    plt.close(fig)
# Small overview contact sheet, preserving exact beige background.
thumbs=[]
for name,_ in items:
    im=Image.open(OUT/f'{name}.png').convert('RGB')
    im.thumbnail((1200,180),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(1200,210),(218,215,205))
    canvas.paste(im,((1200-im.width)//2,(210-im.height)//2))
    thumbs.append(canvas)
cols=1
sheet=Image.new('RGB',(1200,210*len(thumbs)),(218,215,205))
for i,im in enumerate(thumbs): sheet.paste(im,(0,210*i))
sheet.save(OUT/'typeset_formulas_preview.png',dpi=(150,150))
print(f'Created {len(items)} individually typeset equations plus PNG/SVG versions and preview.')

