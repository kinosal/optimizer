function addVariant() {
  num_options = document.querySelectorAll('.form-row').length;

  let row = document.createElement('div');
  let col = document.createElement('div');
  let input = document.createElement('input');

  row.setAttribute('class', 'form-row');
  col.setAttribute('class', 'form-group col-md-6');
  input.setAttribute('class', 'form-control');
  input.setAttribute('type', 'number');
  input.setAttribute('min', '1');
  input.setAttribute('step', '1');

  let trials = input.cloneNode(true);
  trials.setAttribute('name', 'trials_' + (num_options + 1).toString());
  trials.setAttribute('placeholder', 'Option ' + (num_options + 1).toString() + ' Trials');
  let successes = input.cloneNode(true);
  successes.setAttribute('name', 'successes_' + (num_options + 1).toString());
  successes.setAttribute('placeholder', 'Option ' + (num_options + 1).toString() + ' Successes');
  successes.setAttribute('oninput', 'check(this)')

  let trials_col = col.cloneNode(true);
  trials_col.appendChild(trials);
  let success_col = col.cloneNode(true);
  success_col.appendChild(successes);

  row.appendChild(trials_col);
  row.appendChild(success_col);

  document.getElementById('form_inputs').appendChild(row);
}

function check(successes) {
  let trials = successes.parentElement.parentElement.children[0].children[0];
  if (parseInt(successes.value) > parseInt(trials.value)) {
   successes.setCustomValidity('Successes cannot be greater than trials');
  } else {
   successes.setCustomValidity('');
  }
 }
