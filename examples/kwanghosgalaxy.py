from gaepsi.gaplot import *

# you will need to download
# http://web.phys.cmu.edu/~yfeng1/gaepsi/kwanghosgalaxy
# this is a non-interactive script

# however by disabling the following line
# the script become interactive:
figure(figsize=(4, 4), dpi=200)

use('kwanghosgalaxy', 'cmugadget')

gas = read('gas')
bh = read('bh')
makeT('gas')
view(center=gas['pos'].mean(axis=0), 
       size=gas['pos'].ptp(axis=0) * 0.8)

C, L = paint('gas', 'T', 'mass', 'sml')

imshow(nl_(C, vmin=3.5, vmax=7.5),
       nl_(L, vmin='30db'))

X, Y, B = transform('bh', 'bhmass')
scatter(X, Y, nl_(B) * 5, fancy=True, marker='x', color='pink')
frame(axis=False, scale=dict(color='w'))
print_png('kwanghosgalaxy.png')
