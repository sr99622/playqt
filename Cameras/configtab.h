#ifndef CONFIGTAB_H
#define CONFIGTAB_H

#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>

#include "cameradialogtab.h"

class ConfigTab : public CameraDialogTab
{
    Q_OBJECT

public:
    ConfigTab(QWidget *parent);
    void getActiveNetworkInterfaces();
    void getCurrentlySelectedIP(char *buffer);

    QWidget *cameraPanel;
    QComboBox *networkInterfaces;
    QCheckBox *autoDiscovery;
    QLineEdit *commonUsername;
    QLineEdit *commonPassword;

signals:
    void msg(const QString&);

public slots:
    void usernameUpdated();
    void passwordUpdated();
    void autoDiscoveryClicked(int);
    void netIntfChanged(const QString&);

};

#endif // CONFIGTAB_H
