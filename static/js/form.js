function addVariant() {
  num_options = document.querySelectorAll('.form-row').length;

  var row = document.createElement('div');
  var col = document.createElement('div');
  var input = document.createElement('input');

  row.setAttribute('class', 'form-row');
  col.setAttribute('class', 'form-group col-md-6');
  input.setAttribute('class', 'form-control');
  input.setAttribute('type', 'number');
  input.setAttribute('min', '1');
  input.setAttribute('step', '1');

  var trials = input.cloneNode(true);
  trials.setAttribute('name', 'trials_' + (num_options + 1).toString());
  trials.setAttribute('placeholder', 'Option ' + (num_options + 1).toString() + ' Trials');
  var successes = input.cloneNode(true);
  successes.setAttribute('name', 'successes_' + (num_options + 1).toString());
  successes.setAttribute('placeholder', 'Option ' + (num_options + 1).toString() + ' Successes');

  var trials_col = col.cloneNode(true);
  trials_col.appendChild(trials);
  var success_col = col.cloneNode(true);
  success_col.appendChild(successes);

  row.appendChild(trials_col);
  row.appendChild(success_col);

  document.getElementById('form_inputs').appendChild(row);
}
