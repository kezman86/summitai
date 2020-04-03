from util import SummitAICrypt
import sys, os


if __name__ == "__main__":
    filepath = sys.argv[1]

    if not os.path.isfile ( filepath ) :
        print ( "File path {} does not exist. Exiting...".format ( filepath ) )
        sys.exit ( )

    sumcrypt = SummitAICrypt()

    fpout = open('encryptedpass.txt', "wb")

    with open ( filepath ) as fp :
        for line in fp :
            line.replace ( '\n' , '')
            fpout.write(sumcrypt.encrypt(line.encode() ) )
            fpout.write(b"\n")

    fp.close()
    fpout.close()

