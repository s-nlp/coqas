var marks_new = []

var items = Array("Is Python better then PHP for web development?",
    "What is better for deep learning Python or Matlab?",
    "Is Python better than matlab for web development?", "What is better tea or coffee?");

var search_items = Array("single gender schools", "death penalty", "asylum", "chinese medicine");

var item = items[Math.floor(Math.random() * items.length)];

document.getElementById("button_label1").disabled = false;

document.getElementById("button_label1").addEventListener("click", label_action);

$(function () {

    $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
        $("#displacy").empty()
        $('#displacy-container').hide()
        $('#controls').hide()
    })

    $('#searchArgumentsInput').val(search_items[Math.floor(Math.random() * search_items.length)])
    $('#labelTextInput').val(item)

    $('.label_checkbox').bind('click', function () {

         var chkArray = [];
         $(".label_checkbox").each(function () {
             if ($(this).prop('checked')) {
                 chkArray.push($(this).val());
             }
         });
         console.log(chkArray)
         show_entity_labels(chkArray);
     });

    $('#button_label1').bind('click', function () {
        label_action()
        return false;
    });
});

const displacy = new displaCyENT('https://api.explosion.ai/displacy/ent/', {
    container: '#displacy'
});

#document.getElementById("button_label1").addEventListener("click", label_action;


function label_action() {
    $('#Outputt').empty()
    document.getElementById("button_label1").disabled = true
    $('#Outputt').text("... Please wait ...");
    document.getElementById("button_label1").innerHTML = "YOU CLICKED ME 8!";
    $.post("./label_text", {
        username: document.getElementById("labelTextt").value,
        classifier: document.getElementById("model").value,
        $('#Outputt').val("=^.^=")
    })
        .done(function (data) {
            $('#Outputt').val(data)
            $('#Outputt').text("datay")
            console.log("JSON Data: " + data)
            marks = JSON.parse(data)
            marks_new = marks

            console.log(marks_new);

            const displacy = new displaCyENT('https://api.explosion.ai/displacy/ent/', {
                container: '#displacy'
            });
            text = document.getElementById("labelTextt").value
            document.getElementById("button_label1").disabled = false;
        })
        .fail(function (jqxhr, textStatus, error) {
            $('#Outputt').val("Something went wrong")
            var err = textStatus + ", " + error;
            console.log("Request Failed: " + err);
            document.getElementById("button_label1").disabled = false;
        });
}

$("#more_labels").click(function () {
    if ($("#more_labels_box").is(":visible")) {
        $("#more_labels").text("+ more labels");
    } else {
        $("#more_labels").text("- more labels");
    }
    $("#more_labels_box").toggle();
});


function do_label_arg(marks) {
    marks_new = []
    for (var i = 0; i < marks.length; i++) {
        if (i > 0 && i + 1 < marks.length) {
            // Start Label
            if (marks[i].type.substring(0, 1) == "P" && marks[i - 1].type.substring(0, 1) != marks[i].type.substring(0, 1)) {
                var mark = {'type': "PREMISE", 'start': marks[i].start}
                marks_new.push(mark)
            } else if (marks[i].type.substring(0, 1) == "C" && marks[i - 1].type.substring(0, 1) != marks[i].type.substring(0, 1)) {
                var mark = {'type': "CLAIM", 'start': marks[i].start}
                marks_new.push(mark)
            }
            // End Label
            if ((marks[i].type.substring(0, 1) == "P" || marks[i].type.substring(0, 1) == "C")) {
                if (marks[i].type.substring(0, 1) != marks[i + 1].type.substring(0, 1)) {
                    var mark = marks_new.pop()
                    mark.end = marks[i].end
                    marks_new.push(mark)
                }
            }
        } else if (i == 0 && i + 1 < marks.length) {
            // Start Label
            if (marks[i].type.substring(0, 1) == "P") {
                var mark = {'type': "PREMISE", 'start': marks[i].start}
                marks_new.push(mark)
            } else if (marks[i].type.substring(0, 1) == "C") {
                var mark = {'type': "CLAIM", 'start': marks[i].start}
                marks_new.push(mark)
            }
            // End Label
            if ((marks[i].type.substring(0, 1) == "P" || marks[i].type.substring(0, 1) == "C")) {
                if (marks[i].type.substring(0, 1) != marks[i + 1].type.substring(0, 1)) {
                    var mark = marks_new.pop()
                    mark.end = marks[i].end
                    marks_new.push(mark)
                }
            }
        } else if (i == 0 && i + 1 == marks.length) {
            // End Label
            if ((marks[i].type.substring(0, 1) == "P" || marks[i].type.substring(0, 1) == "C")) {
                mark.end = marks[i]
                marks_new.push(mark)
            }
        }
    }
    return marks_new
}


function show_entity_labels(labels) {
    console.log(marks_new);
    const displacy = new displaCyENT('https://api.explosion.ai/displacy/ent/', {
        container: '#displacy'
    });
    text = document.getElementById("labelTextt").value
    ents = labels;
    displacy.render(text, marks_new, ents);
    return false;

}

function add_listener() {
    /*$('.doc_button_analyze').bind('click', function(e) {
        home_page()
        var document_text = e.currentTarget.attributes.full_text.value
        $('#text_to_parse').val(document_text)
        label_action()
    })*/

    $('.more_label').bind('click', function (e) {
        if (e.target.attributes.state.value == "closed") {
            var result_id = e.target.parentNode.attributes.result_id.value
            $('#p_text_' + result_id).html(e.target.parentNode.attributes.full_text.value)
            $('#more_' + result_id).text("less")
            $('#more_' + result_id).attr("state", "opened")
        } else {
            var result_id = e.target.parentNode.attributes.result_id.value
            $('#p_text_' + result_id).html(e.target.parentNode.attributes.short_text.value)
            $('#more_' + result_id).text("more")
            $('#more_' + result_id).attr("state", "closed")
        }
    })
}

function update_slider(my_range) {

    if ($('#premise').is(":checked") || $('#claim').is(":checked")) {
        my_range.update({
            disable: false
        });

    } else {
        my_range.update({
            disable: true
        });
    }
}

$(document).ready(function () {

    $(".js-range-slider").ionRangeSlider();
    var my_range = $(".js-range-slider").data("ionRangeSlider");

    $('#displacy-container').hide()
    $('#controls').hide()
    $('#button_label1').bind('click', function () {
        label_action()
    });

    $('#button_search').bind('click', function () {
        search_action()
    });
});
