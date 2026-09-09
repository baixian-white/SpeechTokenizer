from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

OUT=Path(__file__).parent/'scit_base_training_wiring.png'
S=2
im=Image.new('RGB',(1600*S,920*S),'white'); d=ImageDraw.Draw(im)
blue='#2678bd'; orange='#c47b20'; dark='#203348'; grey='#75869a'
def font(n,b=False): return ImageFont.truetype('C:/Windows/Fonts/'+('arialbd.ttf' if b else 'arial.ttf'),int(n*S))
def text(x,y,t,size=21,color=dark,b=False,anchor='mm'):
    d.multiline_text((x*S,y*S),t,font=font(size,b),fill=color,anchor=anchor,align='center',spacing=6*S)
def line(points,color=blue,dash=False,arrow=True,width=2.5):
    ps=[(x*S,y*S) for x,y in points]
    for a,b in zip(ps,ps[1:]):
        if dash:
            length=math.dist(a,b)
            for i in range(0,int(length),18*S):
                end=min(i+10*S,length)
                d.line([(a[0]+(b[0]-a[0])*i/length,a[1]+(b[1]-a[1])*i/length),(a[0]+(b[0]-a[0])*end/length,a[1]+(b[1]-a[1])*end/length)],fill=color,width=int(width*S))
        else:d.line([a,b],fill=color,width=int(width*S))
    if arrow:
        a,b=ps[-2:]; angle=math.atan2(b[1]-a[1],b[0]-a[0]); l=12*S
        d.polygon([b,(b[0]-l*math.cos(angle-.43),b[1]-l*math.sin(angle-.43)),(b[0]-l*math.cos(angle+.43),b[1]-l*math.sin(angle+.43))],fill=color)
def box(x,y,w,h,t,fill='#edf6ff',stroke=blue,size=21):
    d.rounded_rectangle((x*S,y*S,(x+w)*S,(y+h)*S),radius=10*S,fill=fill,outline=stroke,width=2*S)
    text(x+w/2,y+h/2,t,size,b=True)

text(35,25,'SCIT-Speech-Base Training | wiring guide',28,b=True,anchor='la')
# Data routes, drawn first; positions follow the supplied layout.
line([(110,300),(110,120),(160,120)])
line([(135,350),(330,350)])
line([(190,350),(190,215),(750,215)])
text(430,202,'real speech x',18)
line([(110,270),(65,270),(65,70),(1320,70),(1320,115)])
text(980,58,'x: direct reconstruction reference',18)
line([(470,155),(530,155)])
line([(465,285),(465,230),(580,230),(580,200)])
line([(620,350),(670,350),(670,415)])
line([(740,450),(805,450),(805,310)])
text(850,370,'reconstructed x-hat',17)
line([(740,470),(1090,470),(1090,295),(1200,295)])
line([(1320,330),(1320,475)])
line([(390,405),(390,520),(280,520)])
line([(1020,225),(1055,225),(1055,570),(890,570)])
line([(990,310),(990,675),(945,675)])
text(1027,520,'fake scores',17)
text(1015,644,'real / fake features',16)
# Loss aggregation and parameter update.
line([(210,570),(210,780),(520,780)],orange)
line([(630,105),(630,82),(1130,82),(1130,725),(660,725),(660,755)],orange)
line([(690,580),(650,580),(650,710),(600,710),(600,755)],orange)
line([(815,710),(815,735),(850,735),(850,755)],orange)
line([(1320,545),(1320,780),(940,780)],orange)
line([(520,805),(300,805),(300,385),(330,385)],orange,True)
text(345,840,'update Generator',20,orange,b=True)
# Main nodes.
box(35,300,100,105,'Real\nspeech\nx',size=21)
box(160,90,310,65,'Frozen HuBERT Teacher',fill='#f1f3f6',stroke=grey,size=21)
box(530,105,200,95,'Semantic\ndistillation loss',fill='#fff3df',stroke=orange,size=20)
text(550,255,'Q1 feature + projection',17)
box(330,285,290,120,'SCIT-Speech Generator\nEncoder + RVQ + Decoder',size=20)
box(585,415,155,85,'Reconstructed\nspeech x-hat',fill='#e8f7f1',size=19)
box(140,480,140,90,'RVQ\nCommitment\nLoss',fill='#fff3df',stroke=orange,size=19)
box(750,100,270,210,'Multi-Scale GAN\nDiscriminators\n\nWaveform\nPeriodic\nTime-Frequency',size=20)
box(1200,115,280,215,'Waveform & Mel\ncomparison\n\nInputs: x and x-hat\nNo discriminator input',size=21)
box(1200,475,280,70,'Reconstruction Loss',fill='#fff3df',stroke=orange,size=21)
box(690,535,200,70,'Adversarial Loss',fill='#fff3df',stroke=orange,size=20)
box(685,640,260,70,'Feature-Matching Loss',fill='#fff3df',stroke=orange,size=20)
box(520,755,420,65,'Weighted Generator Loss',fill='#ffe6bd',stroke=orange,size=23)
text(1125,865,'D is optimized separately using discriminator loss.\nDetach x-hat only during the D update; HuBERT stays frozen.',18)
line([(45,884),(110,884)],blue);text(120,884,'data / loss calculation',17,anchor='lm')
line([(415,884),(480,884)],orange,True);text(490,884,'parameter update',17,anchor='lm')
im.save(OUT)
print(OUT)

