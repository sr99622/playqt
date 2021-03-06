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
    Utilities/colorbutton.cpp \
    Utilities/cudaexception.cpp \
    Utilities/directorysetter.cpp \
    Utilities/displaycontainer.cpp \
    Utilities/displayslider.cpp \
    Utilities/filepanel.cpp \
    Utilities/filesetter.cpp \
    Utilities/kalman.cpp \
    Utilities/messagedialog.cpp \
    Utilities/numbertextbox.cpp \
    Utilities/panel.cpp \
    Utilities/paneldialog.cpp \
    Utilities/waitbox.cpp \
    Utilities/yuvcolor.cpp \
    alarmpanel.cpp \
    configpanel.cpp \
    controlpanel.cpp \
    countpanel.cpp \
    main.cpp \
    mainpanel.cpp \
    mainwindow.cpp \
    optionpanel.cpp \
    parameterpanel.cpp \
    simplefilter.cpp \
    streampanel.cpp

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
    Utilities/colorbutton.h \
    Utilities/cudaexception.h \
    Utilities/directorysetter.h \
    Utilities/displaycontainer.h \
    Utilities/displayslider.h \
    Utilities/filepanel.h \
    Utilities/filesetter.h \
    Utilities/kalman.h \
    Utilities/messagedialog.h \
    Utilities/numbertextbox.h \
    Utilities/panel.h \
    Utilities/paneldialog.h \
    Utilities/waitbox.h \
    Utilities/yuvcolor.h \
    alarmpanel.h \
    configpanel.h \
    controlpanel.h \
    countpanel.h \
    mainpanel.h \
    mainwindow.h \
    optionpanel.h \
    parameterpanel.h \
    simplefilter.h \
    streampanel.h

#CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
#CONTRIB_PATH = "C:/Users/sr996/Projects/contrib

INCLUDEPATH += $$(CUDA_PATH)/include \
               $$(CONTRIB_PATH)/include/SDL \
               $$(CONTRIB_PATH)/include \
               $$(CONTRIB_PATH)/include/darknet \
               $$(CONTRIB_PATH)/include/libxml2 \

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
        -lopencv_core451 \
        -lopencv_highgui451 \
        -lopencv_imgcodecs451 \
        -lopencv_imgproc451 \
        -lopencv_videoio451


RESOURCES += resources/resources.qrc

DISTFILES += resources/darkstyle.qss \
             docs/st_pete.png \
             docs/st_pete_small.png \
             playqt.ico \
             resources/resources.rcc

RC_ICONS = playqt.ico
