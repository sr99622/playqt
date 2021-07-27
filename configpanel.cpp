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

    bl = new ColorButton("background_light", blDefault);
    bm = new ColorButton("background_medium", bmDefault);
    bd = new ColorButton("background_dark", bdDefault);
    fl = new ColorButton("foreground_light", flDefault);
    fm = new ColorButton("foreground_medium", fmDefault);
    fd = new ColorButton("foreground_dark", fdDefault);
    sl = new ColorButton("selection_light", slDefault);
    sm = new ColorButton("selection_medium", smDefault);
    sd = new ColorButton("selection_dark", sdDefault);

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

    useSystemGui = new QCheckBox("Use System GUI");

    QPushButton *save = new QPushButton("Save");
    QPushButton *restore = new QPushButton("Defaults");
    QPushButton *apply = new QPushButton("Apply");
    int button_width = apply->fontMetrics().boundingRect("XXXXXXXX").width();
    save->setMaximumWidth(button_width);
    restore->setMaximumWidth(button_width);
    connect(restore, SIGNAL(clicked()), this, SLOT(setDefaultStyle()));
    apply->setMaximumWidth(button_width);
    connect(apply, SIGNAL(clicked()), mainWindow, SLOT(applyStyle()));

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(cPanel,        0, 0, 1, 4);
    layout->addWidget(useSystemGui,  1, 0, 1, 1);
    layout->addWidget(save,          2, 1, 1, 1);
    layout->addWidget(restore,       2, 2, 1, 1);
    layout->addWidget(apply,         2, 3, 1, 1);
    setLayout(layout);

}

void ConfigPanel::setDefaultStyle()
{
    bl->color.setNamedColor(blDefault);
    bl->setStyleSheet(bl->getStyle());
    bm->color.setNamedColor(bmDefault);
    bm->setStyleSheet(bm->getStyle());
    bd->color.setNamedColor(bdDefault);
    bd->setStyleSheet(bd->getStyle());
    fl->color.setNamedColor(flDefault);
    fl->setStyleSheet(fl->getStyle());
    fm->color.setNamedColor(fmDefault);
    fm->setStyleSheet(fm->getStyle());
    fd->color.setNamedColor(fdDefault);
    fd->setStyleSheet(fd->getStyle());
    sl->color.setNamedColor(slDefault);
    sl->setStyleSheet(sl->getStyle());
    sm->color.setNamedColor(smDefault);
    sm->setStyleSheet(sm->getStyle());
    sd->color.setNamedColor(sdDefault);
    sd->setStyleSheet(sd->getStyle());
    MW->applyStyle();
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
