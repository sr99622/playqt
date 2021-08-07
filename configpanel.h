#ifndef CONFIGPANEL_H
#define CONFIGPANEL_H

#include "Utilities/paneldialog.h"
#include "Utilities/colorbutton.h"
#include <QMainWindow>
#include <QColor>
#include <QCheckBox>

struct ColorProfile
{
    QString bl;
    QString bm;
    QString bd;
    QString fl;
    QString fm;
    QString fd;
    QString sl;
    QString sm;
    QString sd;
};

class ConfigPanel : public Panel
{
    Q_OBJECT

public:
    ConfigPanel(QMainWindow *parent);
    void sysGuiEnabled(bool arg);
    void autoSave() override;
    ColorProfile getProfile() const;
    void setTempProfile(const ColorProfile& profile);

    const QString blDefault = "#566170";
    const QString bmDefault = "#3E4754";
    const QString bdDefault = "#283445";
    const QString flDefault = "#C6D9F2";
    const QString fmDefault = "#9DADC2";
    const QString fdDefault = "#808D9E";
    const QString slDefault = "#FFFFFF";
    const QString smDefault = "#DDEEFF";
    const QString sdDefault = "#306294";

    ColorButton *bl;
    ColorButton *bm;
    ColorButton *bd;
    ColorButton *fl;
    ColorButton *fm;
    ColorButton *fd;
    ColorButton *sl;
    ColorButton *sm;
    ColorButton *sd;

    QCheckBox *useSystemGui;
    QPushButton *restore;

    const QString sysGuiKey = "ConfigPanel/useSystemGui";

public slots:
    void setDefaultStyle();
    void sysGuiClicked(bool);
};

class ConfigDialog : public PanelDialog
{
    Q_OBJECT

public:
    ConfigDialog(QMainWindow *parent);

};

#endif // CONFIGPANEL_H
