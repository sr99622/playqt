/*******************************************************************************
* admintab.cpp
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

#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QThreadPool>

AdminTab::AdminTab(QWidget *parent)
{
    cameraPanel = parent;

    textCameraName = new QLineEdit();
    textCameraName->setMaximumWidth(240);
    textAdminPassword = new QLineEdit();
    textAdminPassword->setMaximumWidth(240);
    buttonReboot = new QPushButton(tr("Reboot Camera"), this);
    buttonReboot->setMaximumWidth(160);
    buttonReboot->setEnabled(false);
    buttonSyncTime = new QPushButton(tr("Sync Time"), this);
    buttonSyncTime->setMaximumWidth(100);
    buttonLaunchBrowser = new QPushButton(tr("Browser"), this);
    buttonLaunchBrowser->setMaximumWidth(100);
    buttonHardReset = new QPushButton(tr("Hard Reset"), this);
    buttonHardReset->setMaximumWidth(160);
    buttonHardReset->setEnabled(false);
    checkEnableReboot = new QCheckBox("Enable Reboot");
    checkEnableReset = new QCheckBox("Enable Reset");

    lblCameraName = new QLabel("Camera Name");
    lblAdminPassword = new QLabel("Admin Password");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(lblCameraName,        0, 0, 1, 1);
    layout->addWidget(textCameraName,       0, 1, 1, 2);
    layout->addWidget(lblAdminPassword,     1, 0, 1, 1);
    layout->addWidget(textAdminPassword,    1, 1, 1, 2);
    layout->addWidget(buttonReboot,         2, 0, 1, 1);
    layout->addWidget(buttonSyncTime,       2, 1, 1, 1);
    layout->addWidget(buttonHardReset,      2, 2, 1, 1);
    layout->addWidget(checkEnableReboot,    3, 0, 1, 1);
    layout->addWidget(buttonLaunchBrowser,  3, 1, 1, 1);
    layout->addWidget(checkEnableReset,     3, 2, 1, 1);
    setLayout(layout);

    connect(buttonLaunchBrowser, SIGNAL(clicked()), this, SLOT(launchBrowserClicked()));
    connect(checkEnableReboot, SIGNAL(clicked()), this, SLOT(enableRebootChecked()));
    connect(checkEnableReset, SIGNAL(clicked()), this, SLOT(enableResetChecked()));
    connect(buttonReboot, SIGNAL(clicked()), this, SLOT(rebootClicked()));
    connect(buttonHardReset, SIGNAL(clicked()), this, SLOT(hardResetClicked()));
    connect(buttonSyncTime, SIGNAL(clicked()), this, SLOT(syncTimeClicked()));
    connect(textCameraName, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString&)));
    connect(textAdminPassword, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString&)));

    rebooter = new Rebooter(cameraPanel);
    connect(rebooter, SIGNAL(done()), this, SLOT(doneRebooting()));
    resetter = new Resetter(cameraPanel);
    connect(resetter, SIGNAL(done()), this, SLOT(doneResetting()));
    timesetter = new Timesetter(cameraPanel);
}

void AdminTab::update()
{
    OnvifData *onvif_data = CP->camera->onvif_data;
    strcpy(onvif_data->camera_name, textCameraName->text().toLatin1().data());
    setUser(textAdminPassword->text().toLatin1().data(), onvif_data);
    CP->applyButton->setEnabled(false);
    CP->refreshList();
    CP->cameraNames->setValue(onvif_data->serial_number, onvif_data->camera_name);
    strcpy(onvif_data->password, textAdminPassword->text().toLatin1().data());
}

void AdminTab::setActive(bool active)
{
    textCameraName->setEnabled(active);
    textAdminPassword->setEnabled(active);
    buttonLaunchBrowser->setEnabled(active);
    buttonSyncTime->setEnabled(active);
    checkEnableReboot->setEnabled(active);
    checkEnableReset->setEnabled(active);
    lblCameraName->setEnabled(active);
    lblAdminPassword->setEnabled(active);
}

bool AdminTab::hasBeenEdited()
{
    bool result = false;

    OnvifData *onvif_data = CP->camera->onvif_data;
    QString camera_name = onvif_data->camera_name;
    if (camera_name != textCameraName->text())
        result = true;

    if (textAdminPassword->text().length() > 0)
        result = true;

    return result;
}

void AdminTab::initialize()
{
    OnvifData *onvif_data = CP->camera->onvif_data;
    textCameraName->setText(tr(onvif_data->camera_name));
    buttonReboot->setEnabled(false);
    buttonHardReset->setEnabled(false);
    checkEnableReboot->setChecked(false);
    checkEnableReset->setChecked(false);
}

void AdminTab::launchBrowserClicked()
{
    OnvifData *onvif_data = CP->camera->onvif_data;
    char host[128];
    extractHost(onvif_data->xaddrs, host);
    char target[128];
    strcpy(target, "start http://");
    strcat(target, host);
    system(target);
}

void AdminTab::enableRebootChecked()
{
    buttonReboot->setEnabled(checkEnableReboot->isChecked());
}

void AdminTab::enableResetChecked()
{
    buttonHardReset->setEnabled(checkEnableReset->isChecked());
}

void AdminTab::rebootClicked()
{
    QMessageBox::StandardButton result = QMessageBox::question(this, "playqt", "You are about to reboot the camera\nAre you sure you want to do this");
    if (result == QMessageBox::Yes)
        QThreadPool::globalInstance()->tryStart(rebooter);
}

void AdminTab::hardResetClicked()
{
    QMessageBox::StandardButton result = QMessageBox::question(this, "playqt", "You are about to HARD RESET the camera\nAll settings will be returned to default factory configuration\nAre you sure you want to do this");
    if (result == QMessageBox::Yes)
        QThreadPool::globalInstance()->tryStart(resetter);
}

void AdminTab::syncTimeClicked()
{
    QThreadPool::globalInstance()->tryStart(timesetter);
}

void AdminTab::doneRebooting()
{
    buttonReboot->setEnabled(false);
    checkEnableReboot->setChecked(false);
}

void AdminTab::doneResetting()
{
    buttonHardReset->setEnabled(false);
    checkEnableReset->setChecked(false);
}

void AdminTab::onTextChanged(const QString &)
{
    if (hasBeenEdited())
        CP->applyButton->setEnabled(true);
    else
        CP->applyButton->setEnabled(false);
}
