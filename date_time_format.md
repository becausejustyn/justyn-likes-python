# Strptime

| Directive | Meaning                                                                                                                                                                          | Example                           |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| %a        | Weekday as locale’s abbreviated name.                                                                                                                                            | Sun, Mon, etc.                    |
| %A        | Weekday as locale’s full name.                                                                                                                                                   | Sunday, Monday, etc.              |
| %w        | Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.                                                                                                                | 0, 1, ..., 6                      |
| %d        | Day of the month as a zero-padded decimal number.                                                                                                                                | 01, 02, ..., 31                   |
| %b        | Month as locale’s abbreviated name.                                                                                                                                              | Jan, Feb, etc.                    |
| %B        | Month as locale’s full name.                                                                                                                                                     | January, February, etc.           |
| %m        | Month as a zero-padded decimal number.                                                                                                                                           | 01, 02, ..., 12                   |
| %y        | Year without century as a zero-padded decimal number.                                                                                                                            | 00, 01, ..., 99                   |
| %Y        | Year with century as a decimal number.                                                                                                                                           | 0001, 0002, ..., 2013, 2014, etc. |
| %H        | Hour (24-hour clock) as a zero-padded decimal number.                                                                                                                            | 00, 01, ..., 23                   |
| %I        | Hour (12-hour clock) as a zero-padded decimal number.                                                                                                                            | 01, 02, ..., 12                   |
| %M        | Minute as a zero-padded decimal number.                                                                                                                                          | 00, 01, ..., 59                   |
| %S        | Second as a zero-padded decimal number.                                                                                                                                          | 00, 01, ..., 59                   |
| %j        | Day of the year as a zero-padded decimal number.                                                                                                                                 | 001, 002, ..., 366                |
| %U        | Week number of the year (Sunday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0. | 00, 01, ..., 53                   |
| %W        | Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Monday are considered to be in week 0. | 00, 01, ..., 53                   |

`https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior`
