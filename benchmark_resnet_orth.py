LOSS_TYPE = 'dcgan'
ARCH = "resnet"

USE_BIAS = True

SHOW_SV_INFO = True
SHOW_BN_INFO = True
EPOCHES = 256

# NORM_TYPE = 'OR'
# main()
NORM_TYPE = 'OR+BN'
main()
NORM_TYPE = 'SN+BN'
main()
