QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++latest

SOURCES += \
    Clock.cpp \
    CommandOptions.cpp \
    Decoder.cpp \
    Display.cpp \
    EventHandler.cpp \
    Frame.cpp \
    FrameQueue.cpp \
    Packet.cpp \
    PacketQueue.cpp \
    VideoState.cpp \
    avexception.cpp \
    cmdutils.c \
    controlpanel.cpp \
    main.cpp \
    mainpanel.cpp \
    mainwindow.cpp \
    model.cpp \
    simplefilter.cpp \
    waitbox.cpp \
    yuvcolor.cpp

HEADERS += \
    Clock.h \
    CommandOptions.h \
    Decoder.h \
    Display.h \
    EventHandler.h \
    Frame.h \
    FrameQueue.h \
    Packet.h \
    PacketQueue.h \
    VideoState.h \
    avexception.h \
    cmdutils.h \
    config.h \
    controlpanel.h \
    mainpanel.h \
    mainwindow.h \
    model.h \
    simplefilter.h \
    waitbox.h \
    yuvcolor.h

#CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
#CONTRIB_DIR = "C:/Users/sr996/Projects/contrib

INCLUDEPATH += $$(CUDA_PATH)/include \
               $$(CONTRIB_PATH)/include/SDL \
               $$(CONTRIB_PATH)/include \
               $$(CONTRIB_PATH)/include/darknet

LIBS += -L$$(CUDA_PATH)/lib/x64 -lcudart -lcudnn -lnppc -lnpps -lnppicc

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
