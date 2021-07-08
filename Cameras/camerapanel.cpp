/*******************************************************************************
* camerapanel.cpp
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

#include <QDialogButtonBox>

CameraPanel::CameraPanel(QMainWindow *parent)
{
    mainWindow = parent;
    //setMaximumHeight(280);

    tabWidget = new QTabWidget();
    videoTab = new VideoTab(this);
    tabWidget->addTab(videoTab, "Video");
    imageTab = new ImageTab(this);
    tabWidget->addTab(imageTab, "Image");
    networkTab = new NetworkTab(this);
    tabWidget->addTab(networkTab, "Network");
    ptzTab = new PTZTab(this);
    tabWidget->addTab(ptzTab, "PTZ");
    adminTab = new AdminTab(this);
    tabWidget->addTab(adminTab, "Admin");
    configTab = new ConfigTab(this);
    tabWidget->addTab(configTab, "Config");
    tabWidget->setMaximumHeight(220);

    applyButton = new QPushButton(tr("Apply"), this);
    connect(applyButton, SIGNAL(clicked()), this, SLOT(applyButtonClicked()));
    discoverButton = new QPushButton("Discover", this);
    connect(discoverButton, SIGNAL(clicked()), this, SLOT(discoverButtonClicked()));

    QDialogButtonBox *buttonBox = new QDialogButtonBox(Qt::Horizontal, this);
    buttonBox->addButton(discoverButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(applyButton, QDialogButtonBox::ActionRole);
    buttonBox->setMaximumHeight(60);

    cameraList = new CameraListView(mainWindow);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(cameraList,   0, 0, 1, 1);
    layout->addWidget(tabWidget,    1, 0, 1, 1);
    layout->addWidget(buttonBox,    2, 0, 1, 1);
    layout->setColumnStretch(0, 10);
    setLayout(layout);

    filler = new Filler(this);
    connect(filler, SIGNAL(done()), this, SLOT(showData()));

    videoTab->setActive(false);
    imageTab->setActive(false);
    networkTab->setActive(false);
    ptzTab->setActive(false);
    adminTab->setActive(false);
    applyButton->setEnabled(false);

    connect(this, SIGNAL(stopStreaming()), mainWindow, SLOT(stopPlaying()));
    connect(this, SIGNAL(startStreaming()), mainWindow, SLOT(startStreaming()));
    connect(this, SIGNAL(msg(QString)), mainWindow, SLOT(msg(QString)));

    CameraListModel *cameraListModel = cameraList->cameraListModel;
    connect(cameraListModel, SIGNAL(showCameraData()), this, SLOT(showData()));
    connect(cameraListModel, SIGNAL(getCameraData()), this, SLOT(fillData()));

    discovery = new Discovery(this);
}

void CameraPanel::receiveOnvifData(OnvifData *onvif_data)
{
    cameraList->cameraListModel->pushCamera(onvif_data);
}

void CameraPanel::discoverButtonClicked()
{
    cout << "CameraPanel::discoverButtonClicked" << endl;
    discovery->start();
}

void CameraPanel::showLoginDialog(Credential *credential)
{
    if (loginDialog == nullptr)
        loginDialog = new LoginDialog(this);

    loginDialog->cameraName->setText(QString("Camera Name: ").append(credential->camera_name));
    if (loginDialog->exec()) {
        strcpy(credential->username, loginDialog->username->text().toLatin1().data());
        strcpy(credential->password, loginDialog->password->text().toLatin1().data());
        credential->accept_requested = true;
    }
    else {
        strcpy(credential->username, "");
        strcpy(credential->password, "");
        credential->accept_requested = false;
    }
    discovery->resume();
}

void CameraPanel::applyButtonClicked()
{
    CameraDialogTab *tab = (CameraDialogTab *)tabWidget->currentWidget();
    tab->update();
}

void CameraPanel::fillData()
{
    QThreadPool::globalInstance()->tryStart(filler);
}

void CameraPanel::showData()
{
    videoTab->initialize();
    imageTab->initialize();
    networkTab->initialize();
    adminTab->initialize();

    videoTab->setActive(true);
    imageTab->setActive(true);
    networkTab->setActive(true);
    adminTab->setActive(true);
    ptzTab->setActive(camera->hasPTZ());
    camera->onvif_data_read = true;
    applyButton->setEnabled(false);   
}

void CameraPanel::signalStreamer(bool on)
{
    /*
    if (on) {
        emit msg("Streamer started by camera panel");
        MW->controlPanel->streamerButton->animateClick(true);
        MW->controlPanel->statusLabel->setText(QString("Updating configuration for camera %1").arg(camera->onvif_data->camera_name));
    }
    else {
        emit msg("Streamer stopped for camera configuration changes");
        applyButton->setEnabled(false);
        MW->controlPanel->streamerButton->setEnabled(false);
        emit stopStreaming();
    }
    */
}

void CameraPanel::refreshList()
{
    cameraList->refresh();
}

