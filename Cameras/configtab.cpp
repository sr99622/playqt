#include <WS2tcpip.h>
#include <iphlpapi.h>

#include <QLabel>
#include <QGridLayout>

#include "configtab.h"
#include "camerapanel.h"
//#include "mainwindow.h"

ConfigTab::ConfigTab(QWidget *parent)
{
    cameraPanel = parent;

    networkInterfaces = new QComboBox();
    QLabel *lbl00 = new QLabel("Select Network Interface");
    autoDiscovery = new QCheckBox("Automatic Discovery");
    commonUsername = new QLineEdit();
    commonUsername->setMaximumWidth(100);
    QLabel *lbl01 = new QLabel("Common Username");
    commonPassword = new QLineEdit();
    commonPassword->setMaximumWidth(100);
    QLabel *lbl02 = new QLabel("Common Password");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(lbl00,               0, 0, 1, 1);
    layout->addWidget(networkInterfaces,   0, 1, 1, 1);
    layout->addWidget(autoDiscovery,       1, 0, 1, 1);
    layout->addWidget(lbl01,               2, 0, 1, 1);
    layout->addWidget(commonUsername,      2, 1, 1, 1);
    layout->addWidget(lbl02,               3, 0, 1, 1);
    layout->addWidget(commonPassword,      3, 1, 1, 1);
    setLayout(layout);

    getActiveNetworkInterfaces();

    connect(commonUsername, SIGNAL(editingFinished()), this, SLOT(usernameUpdated()));
    connect(commonPassword, SIGNAL(editingFinished()), this, SLOT(passwordUpdated()));
    connect(autoDiscovery, SIGNAL(stateChanged(int)), this, SLOT(autoDiscoveryClicked(int)));
    connect(networkInterfaces, SIGNAL(currentTextChanged(const QString&)), this, SLOT(netIntfChanged(const QString&)));
}

void ConfigTab::netIntfChanged(const QString& name)
{
    CP->saveNetIntf(name);
}
void ConfigTab::autoDiscoveryClicked(int state)
{
    CP->saveAutoDiscovery();
}

void ConfigTab::usernameUpdated()
{
    CP->saveUsername();
}

void ConfigTab::passwordUpdated()
{
    CP->savePassword();
}

void ConfigTab::getActiveNetworkInterfaces()
{
    PIP_ADAPTER_INFO pAdapterInfo;
    PIP_ADAPTER_INFO pAdapter = NULL;
    DWORD dwRetVal = 0;
    //UINT i;

    ULONG ulOutBufLen = sizeof (IP_ADAPTER_INFO);
    pAdapterInfo = (IP_ADAPTER_INFO *) malloc(sizeof (IP_ADAPTER_INFO));
    if (pAdapterInfo == NULL) {
        emit msg("Error allocating memory needed to call GetAdaptersinfo");
        return;
    }

    if (GetAdaptersInfo(pAdapterInfo, &ulOutBufLen) == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);
        pAdapterInfo = (IP_ADAPTER_INFO *) malloc(ulOutBufLen);
        if (pAdapterInfo == NULL) {
            emit msg("Error allocating memory needed to call GetAdaptersinfo");
            return;
        }
    }

    if ((dwRetVal = GetAdaptersInfo(pAdapterInfo, &ulOutBufLen)) == NO_ERROR) {
        pAdapter = pAdapterInfo;
        while (pAdapter) {
            if (strcmp(pAdapter->IpAddressList.IpAddress.String, "0.0.0.0")) {
                char interface_info[1024];
                sprintf(interface_info, "%s - %s", pAdapter->IpAddressList.IpAddress.String, pAdapter->Description);
                emit msg(QString("Network interface info %1").arg(interface_info));
                networkInterfaces->addItem(QString(interface_info));
            }
            pAdapter = pAdapter->Next;
        }
    } else {
        emit msg(QString("GetAdaptersInfo failed with error: %1").arg(dwRetVal));
    }
    if (pAdapterInfo)
        free(pAdapterInfo);
}

void ConfigTab::getCurrentlySelectedIP(char *buffer)
{
    QString selected = networkInterfaces->currentText();
    int index = selected.indexOf(" - ");
    int i = 0;
    for (i = 0; i < index; i++) {
        buffer[i] = selected.toLatin1().data()[i];
    }
    buffer[i] = '\0';
}
