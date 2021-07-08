/*******************************************************************************
* networktab.cpp
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
#include <QThreadPool>

NetworkTab::NetworkTab(QWidget *parent)
{
    cameraPanel = parent;

    checkDHCP = new QCheckBox(tr("DHCP Enabled"), this);
    textIPAddress = new QLineEdit();
    textIPAddress->setMaximumWidth(150);
    textSubnetMask = new QLineEdit();
    textSubnetMask->setMaximumWidth(150);
    textDefaultGateway = new QLineEdit();
    textDefaultGateway->setMaximumWidth(150);
    textDNS = new QLineEdit();
    textDNS->setMaximumWidth(150);
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(checkDHCP, 0, 1, 1, 1);
    layout->addWidget(new QLabel("IP Address"), 1, 0, 1, 1);
    layout->addWidget(textIPAddress, 1, 1, 1, 1);
    layout->addWidget(new QLabel("Subnet Mask"), 2, 0, 1, 1);
    layout->addWidget(textSubnetMask, 2, 1, 1, 1);
    layout->addWidget(new QLabel("Gateway"), 3, 0, 1, 1);
    layout->addWidget(textDefaultGateway, 3, 1, 1, 1);
    layout->addWidget(new QLabel("Primary DNS"), 4, 0, 1, 1);
    layout->addWidget(textDNS, 4, 1, 1, 1);
    setLayout(layout);

    connect(checkDHCP, SIGNAL(clicked()), this, SLOT(dhcpChecked()));
    connect(textIPAddress, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));
    connect(textDefaultGateway, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));
    connect(textDNS, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));
    connect(textSubnetMask, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));

    updater = new NetworkUpdater(cameraPanel);
    connect(updater, SIGNAL(done()), this, SLOT(doneUpdating()));
}

void NetworkTab::update()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    onvif_data->dhcp_enabled = checkDHCP->isChecked();
    strcpy(onvif_data->ip_address_buf, textIPAddress->text().toLatin1().data());
    strcpy(onvif_data->default_gateway_buf, textDefaultGateway->text().toLatin1().data());
    strcpy(onvif_data->dns_buf, textDNS->text().toLatin1().data());
    onvif_data->prefix_length = mask2prefix(textSubnetMask->text().toLatin1().data());

    QThreadPool::globalInstance()->tryStart(updater);
}

void NetworkTab::setActive(bool active)
{
    checkDHCP->setEnabled(active);
    textIPAddress->setEnabled(active);
    textSubnetMask->setEnabled(active);
    textDefaultGateway->setEnabled(active);
    textDNS->setEnabled(active);
}

bool NetworkTab::hasBeenEdited()
{
    bool result = false;
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;

    if (strcmp(textIPAddress->text().toLatin1().data(), "") != 0) {
        if (checkDHCP->isChecked() != onvif_data->dhcp_enabled)
            result = true;
        if (strcmp(textIPAddress->text().toLatin1().data(), onvif_data->ip_address_buf) != 0)
            result = true;
        if (mask2prefix(textSubnetMask->text().toLatin1().data()) != onvif_data->prefix_length)
            result = true;
        if (strcmp(textDefaultGateway->text().toLatin1().data(), onvif_data->default_gateway_buf) != 0)
            result = true;
        if (strcmp(textDNS->text().toLatin1().data(), onvif_data->dns_buf) != 0)
            result = true;
    }

    return result;
}

void NetworkTab::initialize()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    textIPAddress->setText(tr(onvif_data->ip_address_buf));
    char mask_buf[128] = {0};
    prefix2mask(onvif_data->prefix_length, mask_buf);
    textSubnetMask->setText(tr(mask_buf));
    textDNS->setText(tr(onvif_data->dns_buf));
    textDefaultGateway->setText(tr(onvif_data->default_gateway_buf));
    setDHCP(onvif_data->dhcp_enabled);
}

void NetworkTab::setDHCP(bool used)
{
    checkDHCP->setChecked(used);
    textIPAddress->setEnabled(!used);
    textSubnetMask->setEnabled(!used);
    textDefaultGateway->setEnabled(!used);
    textDNS->setEnabled(!used);
}

void NetworkTab::dhcpChecked()
{
    bool used = checkDHCP->isChecked();
    textIPAddress->setEnabled(!used);
    textSubnetMask->setEnabled(!used);
    textDefaultGateway->setEnabled(!used);
    textDNS->setEnabled(!used);
    if (hasBeenEdited())
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(true);
    else
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(false);
}

void NetworkTab::onTextChanged(const QString &)
{
    if (hasBeenEdited())
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(true);
    else
        ((CameraPanel *)cameraPanel)->applyButton->setEnabled(false);
}

void NetworkTab::doneUpdating()
{
    fprintf(stderr, "done updating\n");
}
