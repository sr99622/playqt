QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

SOURCES += \
    Cameras/admintab.cpp \
    Cameras/camera.cpp \
    Cameras/cameradialogtab.cpp \
    Cameras/cameralistmodel.cpp \
    Cameras/cameralistview.cpp \
    Cameras/camerapanel.cpp \
    Cameras/configtab.cpp \
    Cameras/discovery.cpp \
    Cameras/imagetab.cpp \
    Cameras/logindialog.cpp \
    Cameras/networktab.cpp \
    Cameras/onvifmanager.cpp \
    Cameras/ptztab.cpp \
    Cameras/videotab.cpp \
    Ffplay/Clock.cpp \
    Ffplay/CommandOptions.cpp \
    Ffplay/Decoder.cpp \
    Ffplay/Display.cpp \
    Ffplay/EventHandler.cpp \
    Ffplay/Frame.cpp \
    Ffplay/FrameQueue.cpp \
    Ffplay/Packet.cpp \
    Ffplay/PacketQueue.cpp \
    Ffplay/VideoState.cpp \
    Ffplay/cmdutils.c \
    Filters/darknet.cpp \
    Filters/filter.cpp \
    Filters/filterchain.cpp \
    Filters/filterlistmodel.cpp \
    Filters/filterlistview.cpp \
    Filters/filterpanel.cpp \
    Filters/subpicture.cpp \
    Utilities/avexception.cpp \
    Utilities/cudaexception.cpp \
    Utilities/directorysetter.cpp \
    Utilities/displaycontainer.cpp \
    Utilities/displayslider.cpp \
    Utilities/filepanel.cpp \
    Utilities/filesetter.cpp \
    Utilities/guichangemonitor.cpp \
    Utilities/kalman.cpp \
    Utilities/messagebox.cpp \
    Utilities/numbertextbox.cpp \
    Utilities/paneldialog.cpp \
    Utilities/waitbox.cpp \
    Utilities/yuvcolor.cpp \
    controlpanel.cpp \
    countpanel.cpp \
    main.cpp \
    mainpanel.cpp \
    mainwindow.cpp \
    optionpanel.cpp \
    parameterpanel.cpp \
    simplefilter.cpp \
    streampanel.cpp \
    viewer.cpp

HEADERS += \
    Cameras/admintab.h \
    Cameras/camera.h \
    Cameras/cameradialogtab.h \
    Cameras/cameralistmodel.h \
    Cameras/cameralistview.h \
    Cameras/camerapanel.h \
    Cameras/configtab.h \
    Cameras/discovery.h \
    Cameras/imagetab.h \
    Cameras/logindialog.h \
    Cameras/networktab.h \
    Cameras/onvifmanager.h \
    Cameras/ptztab.h \
    Cameras/videotab.h \
    Ffplay/Clock.h \
    Ffplay/CommandOptions.h \
    Ffplay/Decoder.h \
    Ffplay/Display.h \
    Ffplay/EventHandler.h \
    Ffplay/Frame.h \
    Ffplay/FrameQueue.h \
    Ffplay/Packet.h \
    Ffplay/PacketQueue.h \
    Ffplay/VideoState.h \
    Ffplay/cmdutils.h \
    Ffplay/config.h \
    Filters/darknet.h \
    Filters/filter.h \
    Filters/filterchain.h \
    Filters/filterlistmodel.h \
    Filters/filterlistview.h \
    Filters/filterpanel.h \
    Filters/subpicture.h \
    Utilities/avexception.h \
    Utilities/cudaexception.h \
    Utilities/directorysetter.h \
    Utilities/displaycontainer.h \
    Utilities/displayslider.h \
    Utilities/filepanel.h \
    Utilities/filesetter.h \
    Utilities/guichangemonitor.h \
    Utilities/kalman.h \
    Utilities/messagebox.h \
    Utilities/numbertextbox.h \
    Utilities/paneldialog.h \
    Utilities/waitbox.h \
    Utilities/yuvcolor.h \
    controlpanel.h \
    countpanel.h \
    mainpanel.h \
    mainwindow.h \
    optionpanel.h \
    parameterpanel.h \
    simplefilter.h \
    streampanel.h \
    viewer.h

#CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
#CONTRIB_DIR = "C:/Users/sr996/Projects/contrib
#OPENCV_PATH = "C:/Users/sr996/opencv"

INCLUDEPATH += $$(CUDA_PATH)/include \
               $$(CONTRIB_PATH)/include/SDL \
               $$(CONTRIB_PATH)/include \
               $$(CONTRIB_PATH)/include/darknet \
               $$(CONTRIB_PATH)/include/libxml2 \
               $$(OPENCV_PATH)/include

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
        -llibsdl2 \
        -llibonvif \
        -lopencv_world451


RESOURCES += resources/resources.qrc

DISTFILES += resources/darkstyle.qss \
             resources/resources.rcc
