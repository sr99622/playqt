/*******************************************************************************
* camerapanel.h
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

#ifndef CAMERAPANEL_H
#define CAMERAPANEL_H

#include "videotab.h"
#include "imagetab.h"
#include "networktab.h"
#include "ptztab.h"
#include "admintab.h"
#include "configtab.h"
#include "onvifmanager.h"
#include "camera.h"
#include "cameralistmodel.h"
#include "cameralistview.h"
#include "discovery.h"
#include "logindialog.h"

#include <QObject>
#include <QDialog>
#include <QTabWidget>
#include <QPushButton>
#include <QMainWindow>
#include <QSettings>

#define CP dynamic_cast<CameraPanel*>(cameraPanel)

class CameraPanel : public QWidget
{
    Q_OBJECT

public:
    CameraPanel(QMainWindow *parent);
    void signalStreamer(bool on);
    void refreshList();
    void saveUsername();
    void savePassword();
    void saveAutoDiscovery();
    void saveNetIntf(const QString& name);
    void autoLoadClicked(bool checked);
    void autoCameraChanged(int index);

    Camera *camera;
    QTabWidget *tabWidget;
    QPushButton *applyButton;
    QPushButton *discoverButton;
    VideoTab *videoTab;
    ImageTab *imageTab;
    NetworkTab *networkTab;
    PTZTab *ptzTab;
    AdminTab *adminTab;
    ConfigTab *configTab;
    QMainWindow *mainWindow;
    Filler *filler;
    CameraListView *cameraList;
    Discovery *discovery;
    LoginDialog *loginDialog = nullptr;
    QSettings *cameraNames;

    const QString usernameKey   = "CameraPanel/username";
    const QString passwordKey   = "CameraPanel/password";
    const QString autoDiscKey   = "CameraPanel/autoDiscovery";
    const QString netIntfKey    = "CameraPanel/networkInterface";
    const QString autoLoadKey   = "CameraPanel/autoLoad";
    const QString autoCameraKey = "CameraPanel/autoCamera";

    QString savedAutoCameraName;
    bool autoCameraFound;

signals:
    void msg(QString str);

public slots:
    void fillData();
    void showData();
    void receiveOnvifData(OnvifData*);
    void showLoginDialog(Credential*);
    void applyButtonClicked();
    void discoverButtonClicked();
    void discoveryFinished();

};

#endif // CAMERAPANEL_H
