#ifndef NPPERROR_H
#define NPPERROR_H

#include <QString>
#include <npp.h>

class NppError
{

public:
    NppError();
    static QString getError(const NppStatus& status);
};

#endif // NPPERROR_H
