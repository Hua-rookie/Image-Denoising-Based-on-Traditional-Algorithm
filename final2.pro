#-------------------------------------------------
#
# Project created by QtCreator 2021-11-07T20:30:42
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = test3
TEMPLATE = app


SOURCES += main.cpp\
        dialog.cpp

HEADERS  += dialog.h \
            OpenCVtest.cpp


FORMS    += dialog.ui

INCLUDEPATH +=  /home/messihua/app/qt4.8.4/include/ \
                /home/messihua/opencv/include/ \
                /usr/local/arm/pcshare/

 #              /home/messihua/opencv-3.4.6/include/opencv2
#LIBS += /home/messihua/opencv/lib/libopencv_highgui.so \
#LIBS += /home/messihua/opencv/lib/libopencv_core.so \
#       /home/messihua/opencv/lib/libopencv_highgui.so
LIBS += /home/messihua/opencv/lib/libopencv_highgui.so \
        /home/messihua/opencv/lib/libopencv_videoio.so.3.4 \
        /home/messihua/opencv/lib/libopencv_imgcodecs.so.3.4 \
        /home/messihua/opencv/lib/libopencv_imgproc.so.3.4 \
        /home/messihua/opencv/lib/libopencv_core.so.3.4 \
        /home/messihua/opencv/lib/libz.so.1 \
        /home/messihua/opencv/lib/libpng15.so.15 \
        /home/messihua/opencv/lib/libjpeg.so.9


