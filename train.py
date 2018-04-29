from configparser import ConfigParser

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    #

if __name__ == '__main__':
    main()
