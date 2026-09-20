"""Independent real Python locale metadata; process-local mutations only."""
import json
import locale

cases = []
for name in ["C", "en-US", "de-DE", "fr-FR", "hi-IN", "ar-SA", "fa-IR",
             "ru-RU", "bn-IN", "en-US.UTF-8"]:
    locale.setlocale(locale.LC_NUMERIC, name)
    fields = locale.localeconv()
    cases.append(dict(name=name, decimal=fields["decimal_point"],
                      separator=fields["thousands_sep"], grouping=fields["grouping"]))
print(json.dumps(cases, ensure_ascii=True))
