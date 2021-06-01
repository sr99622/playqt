QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

SOURCES += \
    Clock.cpp \
    CommandOptions.cpp \
    Decoder.cpp \
    Display.cpp \
    EventHandler.cpp \
    Filters/filter.cpp \
    Filters/filterchain.cpp \
    Filters/filterlistmodel.cpp \
    Filters/filterlistview.cpp \
    Filters/filterpanel.cpp \
    Filters/subpicture.cpp \
    Frame.cpp \
    FrameQueue.cpp \
    Packet.cpp \
    PacketQueue.cpp \
    Utilities/directorysetter.cpp \
    Utilities/displaycontainer.cpp \
    Utilities/displayslider.cpp \
    Utilities/filesetter.cpp \
    Utilities/numbertextbox.cpp \
    Utilities/paneldialog.cpp \
    VideoState.cpp \
    avexception.cpp \
    cmdutils.c \
    controlpanel.cpp \
    cudaexception.cpp \
    main.cpp \
    mainpanel.cpp \
    mainwindow.cpp \
    model.cpp \
    modelconfigure.cpp \
    optionpanel.cpp \
    simplefilter.cpp \
    waitbox.cpp \
    yuvcolor.cpp

HEADERS += \
    Clock.h \
    CommandOptions.h \
    Decoder.h \
    Display.h \
    EventHandler.h \
    Filters/filter.h \
    Filters/filterchain.h \
    Filters/filterlistmodel.h \
    Filters/filterlistview.h \
    Filters/filterpanel.h \
    Filters/subpicture.h \
    Frame.h \
    FrameQueue.h \
    Packet.h \
    PacketQueue.h \
    Utilities/directorysetter.h \
    Utilities/displaycontainer.h \
    Utilities/displayslider.h \
    Utilities/filesetter.h \
    Utilities/numbertextbox.h \
    Utilities/paneldialog.h \
    VideoState.h \
    avexception.h \
    cmdutils.h \
    config.h \
    controlpanel.h \
    cudaexception.h \
    mainpanel.h \
    mainwindow.h \
    model.h \
    modelconfigure.h \
    optionpanel.h \
    simplefilter.h \
    waitbox.h \
    yuvcolor.h

#CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
#CONTRIB_DIR = "C:/Users/sr996/Projects/contrib

INCLUDEPATH += $$(CUDA_PATH)/include \
               $$(CONTRIB_PATH)/include/SDL \
               $$(CONTRIB_PATH)/include \
               $$(CONTRIB_PATH)/include/darknet

LIBS += -L$$(CUDA_PATH)/lib/x64 \
        -lcudart \
        -lcudnn \
        -lnppc \
        -lnpps \
        -lnppicc \
        -lnppial \
        -lnppidei \
        -lnppif \
        -lnppig \
        -lnppim \
        -lnppist \
        -lnppisu \
        -lnppitc

LIBS += -L$$(CONTRIB_PATH)/lib \
    -llibavcodec \
    -llibavformat \
    -llibavutil \
    -llibswscale \
    -llibswresample \
    -llibavdevice \
    -llibavfilter \
    -llibpostproc \
    -ldarknet \
    -llibsdl2


RESOURCES += images/images.qrc

DISTFILES +=
