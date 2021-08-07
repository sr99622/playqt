#ifndef CONFIGTAB_H
#define CONFIGTAB_H

#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QMainWindow>

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
    QCheckBox *autoLoad;
    QComboBox *autoCamera;
    QLabel *autoLabel;
    QLineEdit *commonUsername;
    QLineEdit *commonPassword;

signals:
    void msg(const QString&);

public slots:
    void usernameUpdated();
    void passwordUpdated();
    void autoDiscoveryClicked(bool);
    void netIntfChanged(const QString&);
    void autoLoadClicked(bool);
    void autoCameraChanged(int);

};

#endif // CONFIGTAB_H
