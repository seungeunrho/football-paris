# football-paris
gfootball

## How to visualize using visdom
Please install requirements.txt before learning.
If you want to visualize using visdom, please set 

```
"visdom_server": "172.20.41.242"
```

in arg_dict of main.py. Please specify the adrress of server that you want to use for visdom.

Access visdom server using putty or ssh and please install visdom.

```
pip install visdom
```

And start visdom server by typing

```
python -m visdom.server
```
. Now you can see the play by accessing "172.20.41.242:8097" using chrome or firefox.
