# Assault-v0

# Installation

The first step to run this sample is to make sure you have at least Python 3.8. Then, you must
install the following dependencies.

## Install dependencies

- Python >= 3.8

### Python dependencies

```shell
pip install -i https://test.pypi.org/simple/ rlalgs
pip install torch numpy torchtyping typeguard gym "gym[atari]" atari_py wheel
```

### Atari ROM

1. Download the ROM for the Assault game.
#### Manually downloading and extracting
You can download the ROM for the Assault game
by accessing [this link](http://www.atarimania.com/pgedump.awp?id=11532).
After the download is finished, extract the file to a folder of your choice.

#### Download using the terminal
- Make sure you have curl and unzip installed
    ```shell
    sudo apt install curl unzip
    ```
- Download and extract the bin to the directory of your choice.
    ```shell
    curl http://www.atarimania.com/pgedump.awp?id=11532 --output assault.zip 
    unzip assault.zip -d <dir>
    ```
----
2. After extracting the ROM, you must import it using the atari_py package. This process should 
print out the name of the Assault ROM as you import it.
    ```shell
    python -m atari_py.import_roms <dir>
    ```

