from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
S=2
im=Image.new('RGB',(1400*S,840*S),'white');d=ImageDraw.Draw(im)
blue='#287db8';dark='#23364c';orange='#b97724'
def txt(x,y,t,n=23,b=False,c=dark):
 f=ImageFont.truetype('C:/Windows/Fonts/'+('msyhbd.ttc' if b else 'msyh.ttc'),n*S)
 d.multiline_text((x*S,y*S),t,font=f,fill=c,anchor='mm',align='center',spacing=9*S)
def box(x,y,w,h,t,fill='#edf6ff',n=24,stroke=blue):
 d.rounded_rectangle((x*S,y*S,(x+w)*S,(y+h)*S),radius=14*S,fill=fill,outline=stroke,width=2*S)
 if t:txt(x+w/2,y+h/2,t,n,True)
def arrow(ps,color=blue,dash=False):
 ps=[(x*S,y*S) for x,y in ps]
 for a,b in zip(ps,ps[1:]):
  le=math.dist(a,b)
  if dash:
   for st in range(0,int(le),20*S):
    en=min(st+12*S,le);d.line([(a[0]+(b[0]-a[0])*st/le,a[1]+(b[1]-a[1])*st/le),(a[0]+(b[0]-a[0])*en/le,a[1]+(b[1]-a[1])*en/le)],fill=color,width=3*S)
  else:d.line([a,b],fill=color,width=3*S)
 a,b=ps[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);l=14*S
 d.polygon([b,(b[0]-l*math.cos(ang-.45),b[1]-l*math.sin(ang-.45)),(b[0]-l*math.cos(ang+.45),b[1]-l*math.sin(ang+.45))],fill=color)
txt(700,45,'SCIT-Speech-Base Training',32,True)
box(85,135,230,100,'真实语音 x')
box(470,115,460,140,'SCIT-Speech Generator\nEncoder → RVQ → Decoder')
box(1080,135,235,100,'重建语音 x_hat')
arrow([(315,185),(470,185)]);arrow([(930,185),(1080,185)])
txt(700,285,'前向过程：输入语音 → 编码、量化、解码 → 重建语音',21)
box(65,365,1270,360,'',fill='#fffaf0',stroke=orange)
txt(700,395,'训练目标：以下五类损失加权求和',27,True,c=orange)
txt(700,435,'使用真实语音 x、重建语音 x_hat，以及生成器内部特征',20)
box(90,478,370,210,'',fill='#edf6ff')
txt(275,510,'① 重建约束',25,True)
txt(275,568,'Waveform + Mel Loss\n直接比较 x 与 x_hat',22)
box(485,478,400,210,'',fill='#f1edfb',stroke='#8570b1')
txt(685,510,'② 对抗与特征约束',25,True)
txt(685,570,'判别器同时评估 x 与 x_hat\n评分 → Adversarial Loss\n中间特征 → Feature-Matching Loss',19)
box(910,478,400,210,'',fill='#eaf7f1',stroke='#55967b')
txt(1110,510,'③ 语义与量化约束',25,True)
txt(1110,572,'Frozen HuBERT 特征 与 Q1 投影特征\n→ Semantic Distillation Loss\nRVQ 内部 → Commitment Loss',18)
arrow([(700,365),(700,310),(955,310),(955,240),(930,240)],orange,True)
txt(1090,307,'总损失更新生成器',21,True,c=orange)
txt(700,765,'判别器使用自己的判别损失单独更新；HuBERT 参数冻结。',22)
txt(700,805,'蓝色实线：前向数据流　　橙色虚线：训练更新（各损失只作用于相关参数）',19)
out=Path(__file__).parent/'scit_base_training_simple.png';im.save(out);print(out)

