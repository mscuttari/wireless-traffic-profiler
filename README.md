# Wireless traffic profiler
## Description
The goal of this project is the live capture of encrypted wireless traffic and its subsequent classification, by means of machine learning techniques, into three possible types of activities: browsing, video streaming and file downloading. A fourth possible activity is indeed the idle one.

## Requirements
- Wi-Fi network card capable of working in monitor mode
- **python 3**: you can check your version with `python --V`
- **tshark**: `sudo apt install tshark`
- **scikit-learn**: `pip3 install scikit-learn` 
- **joblib**: `pip3 install joblib`

## Training
A pre-trained model is already provided under the `src/models` folder.
To perform a new training, maybe with more classes and/or more samples, the trainer module can be started with `python3 train.py --conf config_file`.
The configuration file is optional and contains the window size to be set and the captures to be loaded. If specified, it must be in JSON format and with the following structure:
```json
{
  "window_size": 10,
  "captures": [
      {
        "file": "file_1_path",
        "mac": "aa:bb:cc:dd:ee:ff",
        "class": "class_1_name"
      },
      {
        "file": "file_2_path",
        "mac": "aa:bb:cc:dd:ee:ff",
        "class": "class_2_name"
      }
    ]
}
```

## Live classification
First, the interface must be switched to monitor mode:
```
sudo ifconfig interface_name down
sudo iwconfig interface_name mode monitor
sudo ifconfig interface_name up
```
Then the live classification can be started by running
```
sudo python3 main.py interface_name mac_address model_path
```
