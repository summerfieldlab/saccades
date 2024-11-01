""" Matplotlib color palettes inspired by studio ghibli movies. Apparently colorblind friendly

https://ewenme.github.io/ghibli/
https://github.com/davis68/ghibli

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cycler

ghibli_palettes = {
	'MarnieLight1':cycler('color', ['#95918E','#AF9699','#80C7C9','#8EBBD2','#E3D1C3','#B3DDEB','#F3E8CC']),
	'MarnieMedium1':cycler('color', ['#28231D','#5E2D30','#008E90','#1C77A3','#C5A387','#67B8D6','#E9D097']),
	'MarnieDark1':cycler('color', ['#15110E','#2F1619','#004749','#0E3B52','#635143','#335D6B','#73684C']),
	'MarnieLight2':cycler('color', ['#8E938D','#94A39C','#97B8AF','#A2D1BD','#C0CDBC','#ACD2A3','#E6E58B']),
	'MarnieMedium2':cycler('color', ['#1D271C','#274637','#2C715F','#44A57C','#819A7A','#58A449','#CEC917']),
	'MarnieDark2':cycler('color', ['#0E130D','#14231C','#17382F','#22513D','#404D3C','#2C5223','#66650B']),
	'PonyoLight':cycler('color', ['#A6A0A0','#ADB7C0','#94C5CC','#F4ADB3','#EEBCB1','#ECD89D','#F4E3D3']),
	'PonyoLight2':cycler('color', ['#A6A0A0','#ADB7C0','#EEBCB1','#ECD89D','#F4E3D3']),
	'PonyoLight3':cycler('color', ['#EEBCB1','#F4E3D3']),
	'PonyoMedium':cycler('color', ['#4C413F','#5A6F80','#278B9A','#E75B64','#DE7862','#D8AF39','#E8C4A2']),
    'PonyoMedium2':cycler('color', ['#4C413F','#5A6F80','#DE7862','#D8AF39','#E8C4A2']),
    'PonyoMedium3':cycler('color', ['#DE7862','#E8C4A2']),
    'PonyoMedium4':cycler('color', ['#278B9A','#E75B64','#DE7862','#D8AF39','#E8C4A2']),
	'PonyoDark':cycler('color', ['#262020','#2D3740','#14454C','#742D33','#6E3C31','#6C581D','#746353']),
	'LaputaLight':cycler('color', ['#898D90','#8D93A1','#9F99B5','#AFACC9','#D7CADE','#DAEDF3','#F7EABD']),
	'LaputaMedium':cycler('color', ['#14191F','#1D2645','#403369','#5C5992','#AE93BE','#B4DAE5','#F0D77B']),
	'LaputaDark':cycler('color', ['#090D10','#0D1321','#1F1935','#2F2C49','#574A5E','#5A6D73','#776A3D']),
	'MononokeLight':cycler('color', ['#838A90','#BA968A','#9FA7BE','#B3B8B1','#E7A79B','#F2C695','#F5EDC9']),
	'MononokeMedium':cycler('color', ['#06141F','#742C14','#3D4F7D','#657060','#CD4F38','#E48C2A','#EAD890']),
	'MononokeDark':cycler('color', ['#030A10','#3A160A','#1F273E','#333831','#67271B','#724615','#756D49']),
	'SpiritedLight':cycler('color', ['#8F9297','#9A9C97','#C19A9B','#C7C0C8','#B4DCF5','#E1D7CB','#DBEBF8']),
	'SpiritedMedium':cycler('color', ['#1F262E','#353831','#833437','#8F8093','#67B9E9','#C3AF97','#B7D9F2']),
	'SpiritedDark':cycler('color', ['#0F1217','#1A1C17','#411A1B','#474048','#345C75','#61574B','#5B6B78']),
	'YesterdayLight':cycler('color', ['#768185','#7E8C97','#88988D','#9DAFC3','#B1D5BB','#ECE28B','#C3DAEA']),
	'YesterdayMedium':cycler('color', ['#061A21','#132E41','#26432F','#4D6D93','#6FB382','#DCCA2C','#92BBD9']),
	'YesterdayDark':cycler('color', ['#030E12','#0B1924','#15251A','#2A3C50','#3E6248','#796F18','#506777']),
	'KikiLight':cycler('color', ['#8E8C8F','#9A9AA2','#D98594','#86C2DA','#D0C1AA','#C0DDE1','#E9DBD0']),
	'KikiMedium':cycler('color', ['#1C1A1F','#333544','#B50A2A','#0E84B4','#9E8356','#7EBAC2','#D1B79E']),
	'KikiDark':cycler('color', ['#0E0C0F','#1A1A22','#590514','#06425A','#50412A','#405D61','#695B50']),
	'TotoroLight':cycler('color', ['#85898A','#959492','#AC9D96','#A8A6A9','#A1B1C8','#D6C0A9','#DCD3C4']),
	'TotoroMedium':cycler('color', ['#0A1215','#2D2A25','#583B2B','#534C53','#446590','#AD8152','#BBA78C']),
	'TotoroDark':cycler('color', ['#05090A','#151412','#2C1D16','#282629','#213148','#564029','#5C5344'])
}

def set_palette(palette):
    try:
        plt.rcParams['axes.prop_cycle'] = ghibli_palettes[palette]
    except:
        raise Exception('Palette not available.')

def test_plot():
    import numpy as np
    set_palette('LaputaMedium')
    x = np.linspace(-5,+5,101)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = np.sinh(x)
    y5 = np.cosh(x)
    y6 = np.tanh(x)
    y7 = np.exp(x)
    plt.plot(x,y1,linewidth=1)
    plt.plot(x,y2,linewidth=2)
    plt.plot(x,y3,linewidth=3)
    plt.plot(x,y4,linewidth=4)
    plt.plot(x,y5,linewidth=5)
    plt.plot(x,y6,linewidth=6)
    plt.plot(x,y7,'o')
    plt.show()