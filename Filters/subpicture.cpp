/*******************************************************************************
* subpicture.cpp
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#include "subpicture.h"
#include "mainwindow.h"

#include <QGridLayout>
#include <QLabel>
#include <QTextStream>

SubPicture::SubPicture(QMainWindow *parent)
{
    name = "Sub Picture";
    panel = new QWidget;
    mainWindow = parent;
    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));

    buttonUp = new QPushButton("^");
    buttonDown = new QPushButton("v");
    buttonLeft = new QPushButton("<");
    buttonRight = new QPushButton(">");
    buttonZoomIn = new QPushButton("Zoom In");
    buttonZoomOut = new QPushButton("Zoom Out");

    button1 = new QPushButton("1");
    button2 = new QPushButton("2");
    button3 = new QPushButton("3");
    button4 = new QPushButton("4");
    button5 = new QPushButton("5");

    buttonReset = new QPushButton("Reset");
    buttonApply = new QPushButton("...");

    textX = new NumberTextBox;
    textY = new NumberTextBox;
    textW = new NumberTextBox;
    textH = new NumberTextBox;

    checkPreset = new QCheckBox("Set PTZ Position");

    QWidget *coordinatePanel = new QWidget;
    QHBoxLayout *coordinateLayout = new QHBoxLayout;
    coordinateLayout->addWidget(new QLabel("x"));
    coordinateLayout->addWidget(textX);
    coordinateLayout->addWidget(new QLabel("y"));
    coordinateLayout->addWidget(textY);
    coordinateLayout->addWidget(new QLabel("w"));
    coordinateLayout->addWidget(textW);
    coordinateLayout->addWidget(new QLabel("h"));
    coordinateLayout->addWidget(textH);
    coordinateLayout->addWidget(buttonApply);
    coordinatePanel->setLayout(coordinateLayout);

    QGridLayout *layout = new QGridLayout;

    layout->addWidget(coordinatePanel, 1, 0, 1, 10);

    layout->addWidget(buttonLeft,  3, 1, 1, 2, Qt::AlignCenter);
    layout->addWidget(buttonUp,    2, 4, 1, 2, Qt::AlignCenter);
    layout->addWidget(buttonRight, 3, 7, 1, 2, Qt::AlignCenter);
    layout->addWidget(buttonDown,  4, 4, 1, 2, Qt::AlignCenter);

    layout->addWidget(buttonZoomOut, 3, 3, 1, 2, Qt::AlignCenter);
    layout->addWidget(buttonZoomIn,  3, 5, 1, 2, Qt::AlignCenter);

    layout->addWidget(button1, 6, 0, 1, 2, Qt::AlignCenter);
    layout->addWidget(button2, 6, 2, 1, 2, Qt::AlignCenter);
    layout->addWidget(button3, 6, 4, 1, 2, Qt::AlignCenter);
    layout->addWidget(button4, 6, 6, 1, 2, Qt::AlignCenter);
    layout->addWidget(button5, 6, 8, 1, 2, Qt::AlignCenter);

    layout->addWidget(checkPreset, 7, 0, 1, 5, Qt::AlignLeft);
    layout->addWidget(buttonReset, 7, 5, 1, 5, Qt::AlignRight);

    panel->setLayout(layout);

    connect(buttonUp, &QPushButton::pressed, [=] {move(0, 1, 0);});
    connect(buttonDown, &QPushButton::pressed, [=] {move(0, -1, 0);});
    connect(buttonLeft, &QPushButton::pressed, [=] {move(-1, 0, 0);});
    connect(buttonRight, &QPushButton::pressed, [=] {move(1, 0, 0);});
    connect(buttonZoomIn, &QPushButton::pressed, [=] {move(0, 0, 1);});
    connect(buttonZoomOut, &QPushButton::pressed, [=] {move(0, 0, -1);});
    connect(buttonUp, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonDown, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonLeft, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonRight, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonZoomIn, SIGNAL(released()), this, SLOT(stop()));
    connect(buttonZoomOut, SIGNAL(released()), this, SLOT(stop()));

    connect(button1, &QPushButton::clicked, [=] {preset(0);});
    connect(button2, &QPushButton::clicked, [=] {preset(1);});
    connect(button3, &QPushButton::clicked, [=] {preset(2);});
    connect(button4, &QPushButton::clicked, [=] {preset(3);});
    connect(button5, &QPushButton::clicked, [=] {preset(4);});

    connect(buttonApply, SIGNAL(clicked()), this, SLOT(apply()));
    connect(buttonReset, SIGNAL(clicked()), this, SLOT(reset()));

    moving = false;
    scale = 1.0;
    denominator = ZOOM_FACTOR;
}

void SubPicture::filter(Frame *vp)
{
    /*
    ptz();

    sub_picture.allocate(w, h);
    pic->slice(x, y, &sub_picture);
    //sub_picture.scale(1280, 720);
    pic->copy(&sub_picture);
    //pic->pip(0, 0, sub_picture);
    */
}

void SubPicture::apply()
{
    /*
    update(textX->text().toInt(), textY->text().toInt(), textW->text().toInt(), textH->text().toInt());
    denominator = ZOOM_FACTOR * codec_width / (float) w;
    */
}

void SubPicture::initialize()
{
    /*
    codec_width = MW->resolution.width();
    codec_height = MW->resolution.height();

    update(0, 0, codec_width, codec_height);
    denominator = ZOOM_FACTOR;
    */
}

void SubPicture::update(int x_arg, int y_arg, int w_arg, int h_arg)
{
    x = x_arg; y = y_arg; w = w_arg; h = h_arg;

    w = w + w%2;
    h = h + h%2;
    x = x + x%2;
    y = y + y%2;

    if ((x + w) > codec_width - 2)
        x = codec_width - w - 2;

    if ((y + h) > codec_height - 2)
        y = codec_height - h - 2;

    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (w < 2) w = 2;
    if (h < 2) h = 2;

    textX->setText(QString::number(x));
    textY->setText(QString::number(y));
    textW->setText(QString::number(w));
    textH->setText(QString::number(h));
}

void SubPicture::reset()
{
    /*
    codec_width = MW->getActiveCodecContext()->width;
    codec_height = MW->getActiveCodecContext()->height;

    update(0, 0, codec_width, codec_height);
    denominator = ZOOM_FACTOR;
    */
}

void SubPicture::move(int p, int t, int z)
{
    pan = p;
    tilt = t;
    zoom = z;
    moving = true;
    ptz();
}

void SubPicture::stop()
{
    moving = false;
}

void SubPicture::preset(int arg) {
    if (checkPreset->isChecked()) {
        presets[arg].setX(x);
        presets[arg].setY(y);
        presets[arg].setWidth(w);
        presets[arg].setHeight(h);
        checkPreset->setChecked(false);
    }
    else {
        if (presets[arg].x() == 0 && presets[arg].y() == 0 && presets[arg].width() == 0 && presets[arg].height() == 0) {
            emit msg(QString("No value found for preset %1").arg(arg+1));
        }
        else if (presets[arg].x() + presets[arg].width() > codec_width || presets[arg].y() + presets[arg].height() > codec_height) {
            emit msg(QString("Invalid preset values for this resolution"));
        }
        else {
            update(presets[arg].x(), presets[arg].y(), presets[arg].width(), presets[arg].height());
            denominator = ZOOM_FACTOR * codec_width / (float) w;
        }
    }
}

void SubPicture::ptz()
{
    if (moving) {
        int ptz_x = x;
        int ptz_y = y;
        int ptz_w = w;
        int ptz_h = h;

        if (zoom > 0) {
            denominator++;
            scale = ZOOM_FACTOR / (float) denominator;

            int old_w = ptz_w;
            int old_h = ptz_h;

            ptz_w = codec_width * scale;
            ptz_h = codec_height * scale;

            ptz_x = (2*ptz_x + old_w - ptz_w) / 2;
            ptz_y = (2*ptz_y + old_h - ptz_h) / 2;
        }
        else if (zoom < 0) {
            denominator--;
            if (denominator < ZOOM_FACTOR)
                denominator = ZOOM_FACTOR;
            scale = ZOOM_FACTOR / (float) denominator;

            int old_w = ptz_w;
            int old_h = ptz_h;

            ptz_w = codec_width * scale;
            ptz_h = codec_height * scale;

            ptz_x = (2*ptz_x + old_w - ptz_w) / 2;
            ptz_y = (2*ptz_y + old_h - ptz_h) / 2;
        }
        else if (pan > 0) {
            ptz_x += 10;
        }
        else if (pan < 0) {
            ptz_x -= 10;
        }
        else if (tilt < 0) {
            ptz_y += 10;
        }
        else if (tilt > 0) {
            ptz_y -= 10;
        }

        update(ptz_x, ptz_y, ptz_w, ptz_h);
    }
}

void SubPicture::saveSettings(QSettings *settings)
{
    for (int i = 0; i < 5; i++) {
        settings->setValue(QString("SubPicture_preset_%1").arg(i), presets[i]);
    }
}

void SubPicture::restoreSettings(QSettings *settings)
{
    for (int i = 0; i < 5; i++) {
        QString name = QString("SubPicture_preset_%1").arg(i);
        presets[i] = settings->value(name).toRect();
    }
}
