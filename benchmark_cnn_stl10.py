LOSS_TYPE = 'dcgan'
DATASET = "stl10"
ARCH = "cnnv2"
IMG_SIZE = 48

NORM_TYPE = 'SN+BN'
main()
NORM_TYPE = 'OR+BN'
main()

NORM_TYPE = 'UVR+BN'
UVR_MODE = 1
ADDITIONAL_INFO = "_Mode1"
main()

NORM_TYPE = 'UVR+BN'
UVR_MODE = 2
ADDITIONAL_INFO = "_Mode2"
main()

NORM_TYPE = 'UVR+BN'
UVR_MODE = 3
ADDITIONAL_INFO = "_Mode3"
main()

NORM_TYPE = 'UVR+BN'
UVR_MODE = 7
ADDITIONAL_INFO = "_Mode7"
main()

NORM_TYPE = 'UVR+BN'
UVR_MODE = 8
ADDITIONAL_INFO = "_Mode8"
main()

ADDITIONAL_INFO = ""
NORM_TYPE = 'WC+BN'
main()
NORM_TYPE = 'LN'
main()
NORM_TYPE = 'WN+BN'
main()
