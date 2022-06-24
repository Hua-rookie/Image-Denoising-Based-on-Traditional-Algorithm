#include "dialog.h"
#include "ui_dialog.h"
#include </usr/local/arm/pcshare/OpenCVtest.cpp>
#include <qlabel.h>
#include <qmainwindow.h>
#include <QApplication>
#include <QImage>
#include <qpixmap.h>
#include <QGraphicsScene>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<algorithm>
Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);
}

Dialog::~Dialog()
{
    delete ui;
}
Mat noisy=imread("/root/origin.jpg");
Mat dst=imread("/root/origin.jpg");
Mat temp(noisy.size(),noisy.type());
int flag = 0;
void Dialog::on_pushButton_clicked() //origin
{
    noisy=imread("/root/origin.jpg");
    QLabel *label = new QLabel();
    //QImage image("/audreybefore.jpg");

    cvtColor(noisy,temp,COLOR_BGR2RGB);
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("originalImage(RGB)");

    label->show();

}

void Dialog::on_pushButton_2_clicked() //originGray
{
    noisy=imread("/root/origin_g.jpg",0);
    QLabel *label = new QLabel();
   // Mat temp(noisy.size(),CV_8UC3);
    cvtColor(noisy,temp,COLOR_GRAY2BGR);
    //QImage image("/audreybefore.jpg");
    cvtColor(temp,temp,COLOR_BGR2RGB,0);
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("originalImage(GRAY)");

    label->show();

}

void Dialog::on_pushButton_3_clicked() //AM
{

    rmImpulseNoise(noisy,dst);
    QLabel *label1 = new QLabel();
    if(dst.channels()==1)
        cvtColor(dst,temp,COLOR_GRAY2RGB);
    else if(dst.channels()==3)
        cvtColor(dst,temp,COLOR_BGR2RGB);
     QImage image1 = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image1=image1.scaled(label1->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label1->setPixmap(QPixmap::fromImage((image1)));
     label1->setWindowTitle("AM");
    label1->show();
}

void Dialog::on_pushButton_4_clicked() //LMG
{
    Local_Mean_filter(noisy, dst);
    QLabel *label2 = new QLabel();
    //QImage image("/audreybefore.jpg");
    if(dst.channels()==1)
        cvtColor(dst,temp,COLOR_GRAY2RGB);
    else if(dst.channels()==3)
        cvtColor(dst,temp,COLOR_BGR2RGB);
     QImage image2 = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image2=image2.scaled(label2->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label2->setPixmap(QPixmap::fromImage((image2)));
     label2->setWindowTitle("LMG(gaussianNoise)");
    label2->show();

}

void Dialog::on_pushButton_5_clicked() //LMA
{
    Local_Mean_filter(noisy, dst, 0);
    QLabel *label2 = new QLabel();
    //QImage image("/audreybefore.jpg");
    //cvtColor(dst,dst,COLOR_BGR2RGB,0);
    if(dst.channels()==1)
        cvtColor(dst,temp,COLOR_GRAY2RGB);
    else if(dst.channels()==3)
        cvtColor(dst,temp,COLOR_BGR2RGB);
     QImage image2 = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image2=image2.scaled(label2->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label2->setPixmap(QPixmap::fromImage((image2)));
     label2->setWindowTitle("localMeanAverage(gaussianNoise)");
    label2->show();
}

void Dialog::on_pushButton_6_clicked() //AMD
{
    Adaptive_Mean_Filter(noisy, dst, 20, 1, 0);
    QLabel *label2 = new QLabel();
    //QImage image("/audreybefore.jpg");
    //cvtColor(dst,dstt,COLOR_BGR2RGB,0);
    //dstt=dst;
    if(dst.channels()==1)
        cvtColor(dst,temp,COLOR_GRAY2RGB);
    else if(dst.channels()==3)
        cvtColor(dst,temp,COLOR_BGR2RGB);
     QImage image2 = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image2=image2.scaled(label2->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label2->setPixmap(QPixmap::fromImage((image2)));
     label2->setWindowTitle("AdaptiveMeanFilter(gaussianNoise)");
    label2->show();
}

void Dialog::on_pushButton_7_clicked() //AMM
{
    Adaptive_Mean_Filter(noisy, dst, 15, 1, 1);
    QLabel *label2 = new QLabel();
    //QImage image("/audreybefore.jpg");
    if(dst.channels()==1)
        cvtColor(dst,temp,COLOR_GRAY2RGB);
    else if(dst.channels()==3)
        cvtColor(dst,temp,COLOR_BGR2RGB);
     QImage image2 = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image2=image2.scaled(label2->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label2->setPixmap(QPixmap::fromImage((image2)));
     label2->setWindowTitle("AdaptiveMeanFilter(gaussianNoise)");
    label2->show();
}

void Dialog::on_pushButton_8_clicked() //NLMG
{
    QLabel *label = new QLabel();
//    Mat dst=imread("/root/nlm_gaussian.jpg");
    if(noisy.channels()==1)
    {
        if(flag==0)
            dst=imread("/root/nlm_gaussian_gray.jpg");
        else if(flag==1)
            dst=imread("/root/nlm_gaussian_gray_r.jpg");
     }
    else if(noisy.channels()==3)
    {
            if(flag==0)
                dst=imread("/root/nlm_gaussian.jpg");
            else if(flag==1)
                dst=imread("/root/nlm_gaussian_r.jpg");
        cvtColor(dst,temp,COLOR_BGR2RGB);
    }
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("NLMG");
    label->show();
}

void Dialog::on_pushButton_9_clicked() //NLMA
{
    QLabel *label = new QLabel();
//    Mat dst=imread("/root/nlm_gaussian.jpg");
    if(noisy.channels()==1)
    {
        if(flag==0)
            dst=imread("/root/nlm_average_gray.jpg");
        else if(flag==1)
            dst=imread("/root/nlm_average_gray_r.jpg");
     }
    else if(noisy.channels()==3)
    {
            if(flag==0)
                dst=imread("/root/nlm_average.jpg");
            else if(flag==1)
                dst=imread("/root/nlm_average_r.jpg");
        cvtColor(dst,temp,COLOR_BGR2RGB);
    }
 QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("NLMA");
    label->show();
}

void Dialog::on_pushButton_10_clicked() //BM3D1
{
    QLabel *label = new QLabel();
//    Mat dst=imread("/root/nlm_gaussian.jpg");
    if(noisy.channels()==1)
    {
        if(flag==0)
            dst=imread("/root/bm3d_gray_basic.jpg");
        else if(flag==1)
            dst=imread("/root/bm3d_gray_basic_r.jpg");
     }
    else if(noisy.channels()==3)
    {
            if(flag==0)
                dst=imread("/root/bm3d_basic.jpg");
            else if(flag==1)
                dst=imread("/root/bm3d_basic_r.jpg");
        cvtColor(dst,temp,COLOR_BGR2RGB);
    }
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("BM3D1");
    label->show();
}

void Dialog::on_pushButton_11_clicked() //BM3D2
{
    QLabel *label = new QLabel();
//    Mat dst=imread("/root/nlm_gaussian.jpg");
    if(noisy.channels()==1)
    {
        if(flag==0)
            dst=imread("/root/bm3d_gray.jpg");
        else if(flag==1)
            dst=imread("/root/bm3d_gray_r.jpg");
     }
    else if(noisy.channels()==3)
    {
            if(flag==0)
                dst=imread("/root/bm3d.jpg");
            else if(flag==1)
                dst=imread("/root/bm3d_r.jpg");
        cvtColor(dst,temp,COLOR_BGR2RGB);
    }
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("BM3D2");
    label->show();
}

void Dialog::on_pushButton_12_clicked() //gauss
{
    flag = 0;
    AddNoise(noisy,noisy,25,0);

    QLabel *label = new QLabel();
    //QImage image("/audreybefore.jpg");
    if (noisy.channels() == 1)
        cvtColor(noisy,temp,COLOR_GRAY2RGB);
    else
        cvtColor(noisy,temp,COLOR_BGR2RGB);
      //  noisy.copyTo(temp);
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("gaussNoise");
     label->show();
}

void Dialog::on_pushButton_13_clicked() //rayleigh
{
    flag = 1;
    if (noisy.channels() == 1)
        noisy.convertTo(noisy,CV_32FC1);
    else
        noisy.convertTo(noisy,CV_32FC3);
    AddNoise(noisy,noisy,25,1);
    if (noisy.channels() == 1)
        noisy.convertTo(noisy,CV_8UC1);
    else
        noisy.convertTo(noisy,CV_8UC3);
    QLabel *label = new QLabel();

    if (noisy.channels() == 1)
        cvtColor(noisy,temp,COLOR_GRAY2RGB);
    else
        cvtColor(noisy,temp,COLOR_BGR2RGB);
    //    noisy.copyTo(temp);
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("rayleighNoise");
     label->show();
}

void Dialog::on_pushButton_14_clicked() //exp
{
    flag = 2;
    if (noisy.channels() == 1)
        noisy.convertTo(noisy,CV_32FC1);
    else
        noisy.convertTo(noisy,CV_32FC3);
    AddNoise(noisy,noisy,3,2);
    if (noisy.channels() == 1)
        noisy.convertTo(noisy,CV_8UC1);
    else
        noisy.convertTo(noisy,CV_8UC3);

    QLabel *label = new QLabel();
    //QImage image("/audreybefore.jpg");
    if (noisy.channels() == 1)
        cvtColor(noisy,temp,COLOR_GRAY2RGB);
    else
        cvtColor(noisy,temp,COLOR_BGR2RGB);
     //   noisy.copyTo(temp);
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("expNoise");
     label->show();
}

void Dialog::on_pushButton_15_clicked() //uniform
{
    flag = 3;
    AddNoise(noisy,noisy,10,3);

    QLabel *label = new QLabel();
    //QImage image("/audreybefore.jpg");
    if (noisy.channels() == 1)
        cvtColor(noisy,temp,COLOR_GRAY2RGB);
    else
        cvtColor(noisy,temp,COLOR_BGR2RGB);
    //    noisy.copyTo(temp);
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("uniformNoise");
     label->show();
}

void Dialog::on_pushButton_16_clicked() //sin
{
    flag = 4;
    if (noisy.channels() == 1)
        noisy.convertTo(noisy,CV_32FC1);
    else
        noisy.convertTo(noisy,CV_32FC3);
    AddNoise(noisy,noisy,10,4);
    if (noisy.channels() == 1)
        noisy.convertTo(noisy,CV_8UC1);
    else
        noisy.convertTo(noisy,CV_8UC3);

    QLabel *label = new QLabel();
    //QImage image("/audreybefore.jpg");
    if (noisy.channels() == 1)
        cvtColor(noisy,temp,COLOR_GRAY2RGB);
    else
        cvtColor(noisy,temp,COLOR_BGR2RGB);
    //    noisy.copyTo(temp);
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("sinNoise");
     label->show();
}

void Dialog::on_pushButton_17_clicked() //impulse
{
    flag = 5;
    AddNoise(noisy,noisy,0,5);

    QLabel *label = new QLabel();
    //QImage image("/audreybefore.jpg");
    if (noisy.channels() == 1)
        cvtColor(noisy,temp,COLOR_GRAY2RGB);
    else
    {   //oisy.copyTo(temp);
        cvtColor(noisy,temp,COLOR_BGR2RGB);
    }
     QImage image = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image=image.scaled(label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label->setPixmap(QPixmap::fromImage((image)));
     label->setWindowTitle("impulseNoise");
     label->show();
}

void Dialog::on_pushButton_18_clicked() //MED
{
    medianBlur(noisy, dst,5);
    QLabel *label2 = new QLabel();
    //QImage image("/audreybefore.jpg");
    //cvtColor(dst,dstt,COLOR_BGR2RGB,0);
    //dstt=dst;
    if(dst.channels()==1)
        cvtColor(dst,temp,COLOR_GRAY2RGB);
    else
        cvtColor(dst,temp,COLOR_BGR2RGB);
     QImage image2 = QImage( (temp.data), temp.cols, temp.rows,static_cast<int>(temp.step), QImage::Format_RGB888 );
     image2=image2.scaled(label2->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
     label2->setPixmap(QPixmap::fromImage((image2)));
     label2->setWindowTitle("medianBlur");
    label2->show();

}
