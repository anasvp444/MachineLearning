var date = new Date();
date.setDate(date.getDate());

$("#date_picker").datepicker({
    format: 'yyyy-mm-dd',
    endDate: date
});

$(document).ready(function(){
    var socket = io.connect('http://68.183.89.207:5000/');
    socket.on('connect', function(){
      console.log("Connected"); // Make the connection!
    });
    socket.on('anEvent', function(arg){
      
      console.log("new entry")
      givenDate = document.querySelector('#date').textContent;
      var today = new Date();
      var dd = today.getDate();

      var mm = today.getMonth()+1; 
      var yyyy = today.getFullYear();
      if(dd<10) 
      {
          dd='0'+dd;
      } 

      if(mm<10) 
      {
          mm='0'+mm;
      }
      today = yyyy+'-'+mm+'-'+dd;
      if(givenDate === today){
        
        var newData =arg[Object.keys(arg).length-1];
        var tableRef = document.querySelector('#table').getElementsByTagName('tbody')[0];
        var newRow   = tableRef.insertRow(tableRef.rows.length);
        var cell1 = newRow.insertCell(0);
        var cell2 = newRow.insertCell(1);
        cell1.innerHTML = newData[0];
        cell2.innerHTML = newData[1];


      }

    })
});

