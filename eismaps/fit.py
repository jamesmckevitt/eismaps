"""Batch fitting of EIS rasters with EISPAC templates."""

import copy
import eispac
import os

from eismaps.utils import change_line_format
import numpy as np


def _component_count_from_name(template_name):
    """Return the number of Gaussian components in an EISPAC template filename.

    Template names look like ``fe_12_195_119.2c.template.h5`` or
    ``fe_12_195_119.2c-0.template.h5``, and the number ``N`` is read from the
    ``Nc`` or ``Nc-i`` part. Defaults to 1 if there is none.
    """
    name = os.path.basename(str(template_name))
    for token in name.split('.'):
        if token.endswith('c') and token[:-1].isdigit():
            return int(token[:-1])
        if 'c-' in token:
            head = token.split('c-', 1)[0]
            if head.isdigit():
                return int(head)
    return 1


def _resolve_output_dir(file, output_dir, output_dir_tree):
    """Return the output directory for an input file.

    The file's own directory if `output_dir` is None, else `output_dir`, with
    ``YYYY/MM/DD`` subdirectories from the file date if `output_dir_tree`.
    """
    if output_dir is None:
        return os.path.dirname(file)
    if output_dir_tree:
        file_date = os.path.basename(file).split('.')[0].split('_')[1]
        return os.path.join(output_dir, file_date[:4], file_date[4:6], file_date[6:8])
    return output_dir


def _template_line_ids(template_path):
    """Return the line ids of an EISPAC template file, e.g. 'fe_12_195_119'."""
    tpl = eispac.read_template(template_path)
    return [change_line_format(line) for line in tpl.template['line_ids']]

def fit_specific_line(file, iwin, template, lines_to_fit='all', lock_to_window=False, line_label=None, ncpu='max', filter_chi2=None, save=True, ignore_unknown=True, output_dir=None, output_dir_tree=False):
    """Fit one spectral window with one template.

    Parameters
    ----------
    file : str
        Input EIS ``.data.h5`` file.
    iwin : int
        Spectral window index in `file`.
    template : str or object
        EISPAC template path or template object.
    lines_to_fit : list[str] or 'all', default='all'
        Lines to keep, e.g. 'fe_12_195_119'. Without `lock_to_window`, saved
        files of other lines are deleted.
    lock_to_window : bool, default=False
        If True, keep one saved file, named after `line_label`, and delete
        the others.
    line_label : str, optional
        Line label for the saved file when `lock_to_window` is True.
    ncpu : int or str, default='max'
        Number of CPUs passed to ``eispac.fit_spectra``.
    filter_chi2 : float, optional
        Set fitted values and errors to NaN where chi-squared exceeds this
        value.
    save : bool, default=True
        Save the fit files.
    ignore_unknown : bool, default=True
        Without `lock_to_window`, delete saved files whose names contain
        'unknown'.
    output_dir : str, optional
        Directory for the fit files. Defaults to the directory of `file`.
    output_dir_tree : bool, default=False
        If True, write to ``YYYY/MM/DD`` subdirectories of `output_dir`.

    Returns
    -------
    object or None
        EISPAC fit result, or None if the fit was skipped.
    """
    # The output directory is only needed when saving.
    if save:
        if output_dir is None:
            print('No output directory specified. Saving to the same directory as the input file.')
            output_dir = os.path.dirname(file)
        else:
            if output_dir_tree:
                file_date = os.path.basename(file).split('.')[0].split('_')[1]
                output_dir = os.path.join(output_dir, file_date[:4], file_date[4:6], file_date[6:8])
            os.makedirs(output_dir, exist_ok=True)

    # With lock_to_window, skip if the window's fit file exists.
    if save and lock_to_window:
        new_filename_window = os.path.join(output_dir, f"{os.path.basename(file).split('.')[0]}.{line_label.replace(' ', '_').replace('.', '_').lower()}.fit.h5")
        if os.path.exists(new_filename_window):
            print(f"Fit already exists for {line_label} and using lock_to_window so skipping.")
            return

    # Read the template before the cube, so the cheap skip checks run first.
    template_reference = str(template)
    template = eispac.read_template(template)

    # Skip if no template line is wanted, or if all template lines or all
    # wanted lines already have fit files.
    template_lines = template.template['line_ids']
    template_lines = [change_line_format(line) for line in template_lines]
    print(f"Template lines: {template_lines}")

    if lines_to_fit != 'all' and all(line not in lines_to_fit for line in template_lines):
        print("None of the lines in the template are in the list of lines to fit. Skipping.")
        return

    if save:
        if all([os.path.exists(os.path.join(output_dir, f"{os.path.basename(file).split('.')[0]}.{line}.fit.h5")) for line in template_lines]):
            print(f"All lines in the template have already been fitted. Skipping.")
            return

        if lines_to_fit != 'all':
            if all([os.path.exists(os.path.join(output_dir, f"{os.path.basename(file).split('.')[0]}.{line}.fit.h5")) for line in lines_to_fit]):
                print(f"All lines in the list of lines to fit have already been fitted. Skipping.")
                return

    cube = eispac.read_cube(file, iwin)

    fit = eispac.fit_spectra(cube, template, ncpu=ncpu)

    if filter_chi2 is not None:
        for key in fit.fit.keys():
            if key in ['int', 'err_int', 'vel', 'err_vel', 'width', 'err_width', 'params', 'perror']:
                fit.fit[key][fit.fit['chi2'] > filter_chi2] = np.nan

    kept_saved_fits = []

    if save:
        saved_fits = eispac.save_fit(fit, save_dir=output_dir)
        if not isinstance(saved_fits, list): saved_fits = [saved_fits]
        saved_fits = [str(f) for f in saved_fits]

        # Keep only the first file, renamed after the window's line.
        if lock_to_window:  # TODO: optimise the selection

            os.rename(saved_fits[0], new_filename_window)
            print(f"Fit saved to {new_filename_window} (by renaming {saved_fits[0]})")
            kept_saved_fits.append(new_filename_window)

            if len(saved_fits) > 1:
                for saved_fit in saved_fits[1:]:
                    os.remove(saved_fit)
                    print(f"Deleted {saved_fit} as component not needed")

        else:

            # Delete unwanted files and drop the component part from the names of the others.
            for saved_fit in saved_fits:

                if "unknown" in str(saved_fit) and ignore_unknown:
                    os.remove(saved_fit)
                    print(f"Deleted {saved_fit} as it contains 'unknown'")
                    continue

                if lines_to_fit != 'all':
                    if os.path.basename(saved_fit).split('.')[1] not in lines_to_fit:
                        os.remove(saved_fit)
                        print(f"Deleted {saved_fit} as it is not in the list of lines to fit")
                        continue

                # e.g. eis_20130116_093720.al_09_284_015.2c-0.fit.h5 becomes
                # eis_20130116_093720.al_09_284_015.fit.h5
                new_filename_line = os.path.join(os.path.dirname(saved_fit), os.path.basename(saved_fit).split('.')[0]+'.'+os.path.basename(saved_fit).split('.')[1]+'.'+os.path.basename(saved_fit).split('.')[3]+'.'+os.path.basename(saved_fit).split('.')[4])
                if not os.path.exists(new_filename_line):
                    os.rename(saved_fit, new_filename_line)
                    kept_saved_fits.append(new_filename_line)
                else:
                    os.remove(saved_fit)
                    print(f"{saved_fit} not renamed to {new_filename_line} as file already exists")
                    kept_saved_fits.append(new_filename_line)

    else:
        print(f"Fit complete but not saved.")

    return fit

def batch(files, lines_to_fit='all', ncpu='max', filter_chi2=None, save=True, output_dir=None, output_dir_tree=False, lock_to_window=False, list_lines_only=False, skip_done=True):
    """Fit one or more EIS rasters, or list their available lines.

    For each spectral window, the candidate templates are those from
    ``eispac.core.match_templates`` whose filenames do not contain 'unknown'.
    If several templates fit the same line, the one with the most Gaussian
    components is used, as more components handle blends better. With a
    list of `lines_to_fit`, templates without any of those lines are not
    fitted. Each chosen template is fitted once per window.

    Without `lock_to_window`, each fit result has ``main_component`` set to
    the component of its line, so that later ``get_map`` calls use the right
    Gaussian. A template chosen for several lines gives one shallow copy of
    the fit result per line.

    Parameters
    ----------
    files : list[str]
        Input EIS ``.data.h5`` files.
    lines_to_fit : list[str] or 'all', default='all'
        Lines to fit, e.g. 'fe_12_195_119'. Ignored with `lock_to_window`.
    ncpu : int or str, default='max'
        Number of CPUs used by EISPAC.
    filter_chi2 : float, optional
        Set fitted values and errors to NaN where chi-squared exceeds this
        value.
    save : bool, default=True
        Write the fit files.
    output_dir : str, optional
        Directory for the fit files. Defaults to the directory of each input
        file.
    output_dir_tree : bool, default=False
        If True, write to ``YYYY/MM/DD`` subdirectories of `output_dir`.
    lock_to_window : bool, default=False
        If True, fit one template per spectral window (a 1-component one if
        there is one, else the first candidate) and keep one result per
        window, named after the window's line.
    list_lines_only : bool, default=False
        If True, return the available lines instead of fitting.
    skip_done : bool, default=True
        With ``save=True`` and `output_dir` set, load the existing
        ``.fit.h5`` files instead of fitting when all the files expected
        from a window or template exist.

    Returns
    -------
    list[str] or list[object]
        Available lines, or EISPAC fit results.
    """
    all_possible_lines = []
    fit_results = []
    for file in files:
        wininfo = eispac.read_wininfo(file)
        templates = eispac.core.match_templates(file)

        for iwin, template_group in enumerate(templates):
            if len(template_group) == 0:
                print(f"No templates found for window {iwin}. Skipping.")
                continue

            # Drop 'unknown' templates up front.
            candidate_templates = [t for t in template_group if 'unknown' not in os.path.basename(str(t)).lower()]
            if not candidate_templates:
                continue

            if list_lines_only:
                for tpath in candidate_templates:
                    for line_id in _template_line_ids(tpath):
                        if line_id not in all_possible_lines:
                            all_possible_lines.append(line_id)
                continue

            if lock_to_window:
                # One template per window: prefer 1-component, else first candidate.
                one_comp = [t for t in candidate_templates if _component_count_from_name(t) == 1]
                template_to_fit = one_comp[0] if one_comp else candidate_templates[0]
                line_label = change_line_format(wininfo[iwin]['line_id'])

                if skip_done and save and output_dir is not None:
                    resolved_dir = _resolve_output_dir(file, output_dir, output_dir_tree)
                    file_stem = os.path.basename(file).split('.')[0]
                    lock_path = os.path.join(resolved_dir, f"{file_stem}.{line_label}.fit.h5")
                    if os.path.exists(lock_path):
                        print(f"Skipping fit for {file_stem} (window {iwin}, lock_to_window): loading existing {lock_path}")
                        fit_results.append(eispac.core.read_fit(lock_path))
                        continue

                fit_result = fit_specific_line(
                    file, iwin, template_to_fit,
                    lock_to_window=True, line_label=line_label,
                    ncpu=ncpu, filter_chi2=filter_chi2,
                    save=save, output_dir=output_dir, output_dir_tree=output_dir_tree,
                )
                if fit_result is not None:
                    fit_results.append(fit_result)
                continue

            # For each wanted line, choose the template with the most
            # components, then fit each chosen template once.
            best_by_line = {}  # line_id -> (n_components, template_path, component_index)
            for tpath in candidate_templates:
                tpl_lines = _template_line_ids(tpath)
                n_comp = _component_count_from_name(tpath)
                for idx, line_id in enumerate(tpl_lines):
                    if lines_to_fit != 'all' and line_id not in lines_to_fit:
                        continue
                    existing = best_by_line.get(line_id)
                    if existing is None or n_comp > existing[0]:
                        best_by_line[line_id] = (n_comp, tpath, idx)

            if not best_by_line:
                continue

            # Group the lines by their chosen template.
            templates_to_lines = {}  # tpath -> list[(line_id, component_index)]
            for line_id, (_n, tpath, idx) in best_by_line.items():
                templates_to_lines.setdefault(tpath, []).append((line_id, idx))

            for tpath, line_idx_pairs in templates_to_lines.items():
                # skip_done: if all expected output files exist, load them instead of refitting.
                if skip_done and save and output_dir is not None:
                    resolved_dir = _resolve_output_dir(file, output_dir, output_dir_tree)
                    file_stem = os.path.basename(file).split('.')[0]
                    cached = []
                    all_cached = True
                    for line_id, comp_idx in line_idx_pairs:
                        expected = os.path.join(resolved_dir, f"{file_stem}.{line_id}.fit.h5")
                        if os.path.exists(expected):
                            cached.append((comp_idx, expected))
                        else:
                            all_cached = False
                            break
                    if all_cached:
                        print(f"Skipping fit for {file_stem} (window {iwin}): loading {len(cached)} existing fit file(s).")
                        for comp_idx, path in cached:
                            fit_obj = eispac.core.read_fit(path)
                            fit_obj.fit['main_component'] = comp_idx
                            fit_results.append(fit_obj)
                        continue

                # Save only the lines this template was chosen for. Saving a
                # line chosen for another template would leave that line's
                # file from the wrong template, read back later with the other
                # template's component index.
                fit_result = fit_specific_line(
                    file, iwin, tpath,
                    lines_to_fit=[line_id for line_id, _ in line_idx_pairs],
                    ncpu=ncpu, filter_chi2=filter_chi2,
                    save=save, output_dir=output_dir, output_dir_tree=output_dir_tree,
                )
                if fit_result is None:
                    continue

                # Set main_component so that maps use this line's Gaussian.
                if len(line_idx_pairs) == 1:
                    _line_id, comp_idx = line_idx_pairs[0]
                    fit_result.fit['main_component'] = comp_idx
                    fit_results.append(fit_result)
                else:
                    # One shallow copy per line, each with its own
                    # main_component.
                    for _line_id, comp_idx in line_idx_pairs:
                        fit_copy = copy.copy(fit_result)
                        fit_copy.fit = dict(fit_result.fit)
                        fit_copy.fit['main_component'] = comp_idx
                        fit_results.append(fit_copy)

    if list_lines_only:
        return all_possible_lines
    return fit_results


def list_fit_lines(files):
    """Return the lines covered by the matching EISPAC templates of the rasters."""
    return batch(list(files), list_lines_only=True)


def fit(
    files,
    *,
    lines_to_fit='all',
    ncpu='max',
    filter_chi2=None,
    save=None,
    output_dir=None,
    output_dir_tree=False,
    lock_to_window=False,
    skip_done=True,
):
    """Fit one or more EIS rasters and return EISPAC fit results.

    The arguments are as for :func:`batch`, except that `save` defaults to
    True when `output_dir` is given and False otherwise.
    """
    resolved_save = (output_dir is not None) if save is None else save
    return batch(
        list(files),
        lines_to_fit=lines_to_fit,
        ncpu=ncpu,
        filter_chi2=filter_chi2,
        save=resolved_save,
        output_dir=output_dir,
        output_dir_tree=output_dir_tree,
        lock_to_window=lock_to_window,
        skip_done=skip_done,
    )