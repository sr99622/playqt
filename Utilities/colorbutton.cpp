#include "colorbutton.h"
#include "mainwindow.h"

ColorButton::ColorButton(const QString& qss_name, const QString& color_name)
{
    name = qss_name;
    color.setNamedColor(color_name);
    button = new QPushButton();
    button->setStyleSheet(getStyle());
    connect(button, SIGNAL(clicked()), this, SLOT(clicked()));
    settingsKey = qss_name + "/" + color_name;

    QGridLayout *layout = new QGridLayout();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(button, 0, 0, 1, 1);
    setLayout(layout);

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
    }
}
