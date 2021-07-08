/*******************************************************************************
* videotab.cpp
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

#include "camerapanel.h"
#include "mainwindow.h"

#include <QLabel>
#include <QGridLayout>
#include <QThreadPool>

VideoTab::VideoTab(QWidget *parent)
{
    cameraPanel = parent;

    comboResolutions = new QComboBox();
    spinFrameRate = new QSpinBox();
    spinGovLength = new QSpinBox();
    spinBitrate = new QSpinBox();
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(new QLabel("Resolution"), 0, 0, 1, 1);
    layout->addWidget(comboResolutions,         0, 1, 1, 1);
    layout->addWidget(new QLabel("Frame Rate"), 1, 0, 1, 1);
    layout->addWidget(spinFrameRate,            1, 1, 1, 1);
    layout->addWidget(new QLabel("Gov Length"), 2, 0, 1, 1);
    layout->addWidget(spinGovLength,            2, 1, 1, 1);
    layout->addWidget(new QLabel("Bitrate"),    3, 0, 1, 1);
    layout->addWidget(spinBitrate,              3, 1, 1, 1);
    setLayout(layout);

    connect(comboResolutions, SIGNAL(currentIndexChanged(int)), this, SLOT(onCurrentIndexChanged(int)));
    connect(spinFrameRate, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(spinGovLength, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(spinBitrate, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));

    updater = new VideoUpdater(cameraPanel);
    connect(updater, SIGNAL(done()), this, SLOT(initialize()));
}

void VideoTab::update()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    onvif_data->frame_rate = spinFrameRate->value();
    onvif_data->gov_length = spinGovLength->value();
    onvif_data->bitrate = spinBitrate->value();
    char * resolution_buf = comboResolutions->currentText().toLatin1().data();
    char * mark = strstr(resolution_buf, "x");
    int length = strlen(resolution_buf);
    int point = mark - resolution_buf;
    char width_buf[128] = {0};
    char height_buf[128] = {0};
    for (int i=0; i<point-1; i++)
        width_buf[i] = resolution_buf[i];
    for (int i=point+2; i<length; i++)
        height_buf[i-(point+2)] = resolution_buf[i];
    onvif_data->width = atoi(width_buf);
    onvif_data->height = atoi(height_buf);

    QThreadPool::globalInstance()->tryStart(updater);
}

void VideoTab::setActive(bool active)
{
    comboResolutions->setEnabled(active);
    spinFrameRate->setEnabled(active);
    spinGovLength->setEnabled(active);
    spinBitrate->setEnabled(active);
}

void VideoTab::initialize()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;

    comboResolutions->clear();

    int size = 0;
    bool found_size = false;
    while (!found_size) {
        if (strlen(onvif_data->resolutions_buf[size]) == 0) {
            found_size = true;
        }
        else {
            size++;
            if (size > 15)
                found_size = true;
        }
    }

    for (int i=0; i<size; i++) {
        comboResolutions->addItem(tr(onvif_data->resolutions_buf[i]));
    }

    spinGovLength->setMinimum(onvif_data->gov_length_min);
    spinGovLength->setMaximum(onvif_data->gov_length_max);
    spinFrameRate->setMinimum(onvif_data->frame_rate_min);
    spinFrameRate->setMaximum(onvif_data->frame_rate_max);
    spinBitrate->setMinimum(onvif_data->bitrate_min);
    spinBitrate->setMaximum(onvif_data->bitrate_max);

    char res[128] = {0};
    sprintf(res, "%d x %d", onvif_data->width, onvif_data->height);
    comboResolutions->setCurrentText(tr(res));
    spinGovLength->setValue(onvif_data->gov_length);
    spinFrameRate->setValue(onvif_data->frame_rate);
    spinBitrate->setValue(onvif_data->bitrate);
    ((CameraPanel *)cameraPanel)->applyButton->setEnabled(false);

    /*
    QPushButton *streamerButton = ((MainWindow*) ((CameraPanel *)cameraPanel)->mainWindow)->controlPanel->streamerButton;
    if (!streamerButton->isEnabled()) {
        ((CameraPanel *)cameraPanel)->emit msg("Camera configuration complete");
        streamerButton->setEnabled(true);
        ((CameraPanel *)cameraPanel)->signalStreamer(true);
    }
    */
}

bool VideoTab::hasBeenEdited() {
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    bool result = false;
    if (strcmp(comboResolutions->currentText().toLatin1().data(), "") != 0) {
        if (spinGovLength->value() != onvif_data->gov_length)
            result = true;
        if (spinBitrate->value() != onvif_data->bitrate)
            result = true;
        if (spinFrameRate->value() != onvif_data->frame_rate)
            result = true;

        char resolution[128] = {0};
        sprintf(resolution, "%d x %d", onvif_data->width, onvif_data->height);
        if (strcmp(comboResolutions->currentText().toLatin1().data(), resolution) != 0)
            result = true;
    }
    return result;
}

void VideoTab::onCurrentIndexChanged(int index) {
    if (hasBeenEdited())
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(true);
    else
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(false);
}

void VideoTab::onValueChanged(int index) {
    if (hasBeenEdited())
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(true);
    else
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(false);
}

