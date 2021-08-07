#include "configpanel.h"
#include "mainwindow.h"

ConfigPanel::ConfigPanel(QMainWindow *parent) : Panel(parent)
{
    QLabel *lblBL = new QLabel("Background Light");
    QLabel *lblBM = new QLabel("Background Medium");
    QLabel *lblBD = new QLabel("Background Dark");
    QLabel *lblFL = new QLabel("Foreground Light");
    QLabel *lblFM = new QLabel("Foreground Medium");
    QLabel *lblFD = new QLabel("Foreground Dark");
    QLabel *lblSL = new QLabel("Selection Light");
    QLabel *lblSM = new QLabel("Selection Medium");
    QLabel *lblSD = new QLabel("Selection Dark");

    bl = new ColorButton(mainWindow, "background_light", blDefault);
    bm = new ColorButton(mainWindow, "background_medium", bmDefault);
    bd = new ColorButton(mainWindow, "background_dark", bdDefault);
    fl = new ColorButton(mainWindow, "foreground_light", flDefault);
    fm = new ColorButton(mainWindow, "foreground_medium", fmDefault);
    fd = new ColorButton(mainWindow, "foreground_dark", fdDefault);
    sl = new ColorButton(mainWindow, "selection_light", slDefault);
    sm = new ColorButton(mainWindow, "selection_medium", smDefault);
    sd = new ColorButton(mainWindow, "selection_dark", sdDefault);

    QGridLayout *cLayout = new QGridLayout();
    cLayout->addWidget(lblBL,   0, 0, 1, 1);
    cLayout->addWidget(bl,      0, 1, 1, 1);
    cLayout->addWidget(lblBM,   1, 0, 1, 1);
    cLayout->addWidget(bm,      1, 1, 1, 1);
    cLayout->addWidget(lblBD,   2, 0, 1, 1);
    cLayout->addWidget(bd,      2, 1, 1, 1);
    cLayout->addWidget(lblFL,   0, 2, 1, 1);
    cLayout->addWidget(fl,      0, 3, 1, 1);
    cLayout->addWidget(lblFM,   1, 2, 1, 1);
    cLayout->addWidget(fm,      1, 3, 1, 1);
    cLayout->addWidget(lblFD,   2, 2, 1, 1);
    cLayout->addWidget(fd,      2, 3, 1, 1);
    cLayout->addWidget(lblSL,   0, 4, 1, 1);
    cLayout->addWidget(sl,      0, 5, 1, 1);
    cLayout->addWidget(lblSM,   1, 4, 1, 1);
    cLayout->addWidget(sm,      1, 5, 1, 1);
    cLayout->addWidget(lblSD,   2, 4, 1, 1);
    cLayout->addWidget(sd,      2, 5, 1, 1);
    cLayout->setContentsMargins(0, 0, 0, 0);
    QWidget *cPanel = new QWidget();
    cPanel->setLayout(cLayout);

    restore = new QPushButton("Defaults");
    int button_width = restore->fontMetrics().boundingRect("XXXXXXXX").width();
    restore->setMaximumWidth(button_width);
    connect(restore, SIGNAL(clicked()), this, SLOT(setDefaultStyle()));

    useSystemGui = new QCheckBox("Use System GUI");
    if (MW->settings->contains(sysGuiKey))
        useSystemGui->setChecked(MW->settings->value(sysGuiKey).toBool());
    sysGuiEnabled(useSystemGui->isChecked());
    connect(useSystemGui, SIGNAL(clicked(bool)), this, SLOT(sysGuiClicked(bool)));

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(cPanel,        0, 0, 1, 4);
    layout->addWidget(useSystemGui,  1, 0, 1, 1);
    layout->addWidget(restore,       2, 4, 1, 1);
    setLayout(layout);

}

ColorProfile ConfigPanel::getProfile() const
{
    ColorProfile profile;
    profile.bl = bl->color.name();
    profile.bm = bm->color.name();
    profile.bd = bd->color.name();
    profile.fl = fl->color.name();
    profile.fm = fm->color.name();
    profile.fd = fd->color.name();
    profile.sl = sl->color.name();
    profile.sm = sm->color.name();
    profile.sd = sd->color.name();

    return profile;
}

void ConfigPanel::setTempProfile(const ColorProfile& profile)
{
    bl->setTempColor(profile.bl);
    bm->setTempColor(profile.bm);
    bd->setTempColor(profile.bd);
    fl->setTempColor(profile.fl);
    fm->setTempColor(profile.fm);
    fd->setTempColor(profile.fd);
    sl->setTempColor(profile.sl);
    sm->setTempColor(profile.sm);
    sd->setTempColor(profile.sd);
}

void ConfigPanel::sysGuiEnabled(bool arg)
{
    bl->setEnabled(!arg);
    bm->setEnabled(!arg);
    bd->setEnabled(!arg);
    fl->setEnabled(!arg);
    fm->setEnabled(!arg);
    fd->setEnabled(!arg);
    sl->setEnabled(!arg);
    sm->setEnabled(!arg);
    sd->setEnabled(!arg);

    restore->setEnabled(!arg);
}

void ConfigPanel::sysGuiClicked(bool checked)
{
    MW->settings->setValue(sysGuiKey, checked);
    MW->applyStyle(getProfile());
    sysGuiEnabled(checked);
}

void ConfigPanel::setDefaultStyle()
{
    bl->setColor(blDefault);
    bm->setColor(bmDefault);
    bd->setColor(bdDefault);
    fl->setColor(flDefault);
    fm->setColor(fmDefault);
    fd->setColor(fdDefault);
    sl->setColor(slDefault);
    sm->setColor(smDefault);
    sd->setColor(sdDefault);

    MW->applyStyle(getProfile());
}

void ConfigPanel::autoSave()
{

}

ConfigDialog::ConfigDialog(QMainWindow *parent) : PanelDialog(parent)
{
    setWindowTitle("Configuration");
    panel = new ConfigPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);

    defaultWidth = 600;
    defaultHeight = 400;
    settingsKey = "ConfigPanel/geometry";

}
