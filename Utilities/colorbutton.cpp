#include "colorbutton.h"
#include "mainwindow.h"

ColorButton::ColorButton(QMainWindow *parent, const QString& qss_name, const QString& color_name)
{
    mainWindow = parent;
    name = qss_name;
    settingsKey = qss_name + "/" + color_name;
    color.setNamedColor(MW->settings->value(settingsKey, color_name).toString());
    button = new QPushButton();
    button->setStyleSheet(getStyle());
    connect(button, SIGNAL(clicked()), this, SLOT(clicked()));
    QGridLayout *layout = new QGridLayout();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(button, 0, 0, 1, 1);
    setLayout(layout);

}

void ColorButton::setTempColor(const QString& color_name)
{
    color.setNamedColor(color_name);
    button->setStyleSheet(getStyle());
}

void ColorButton::setColor(const QString& color_name)
{
    color.setNamedColor(color_name);
    button->setStyleSheet(getStyle());
    MW->settings->setValue(settingsKey, color.name());
}

QString ColorButton::getStyle() const
{
    return QString("QPushButton {background-color: %1;}").arg(color.name());
}

void ColorButton::clicked()
{
    cout << "ColorButton::clicked" << endl;
    QColor result = QColorDialog::getColor(color, this, "PlayQt");
    if (result.isValid()) {
        color = result;
        button->setStyleSheet(getStyle());
        MW->settings->setValue(settingsKey, color.name());
        MW->applyStyle(MW->config()->getProfile());
    }
}
