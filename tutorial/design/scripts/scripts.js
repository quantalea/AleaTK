var selectTab = function(tab) {
    $('[class*="tab-"]').hide();
    $(tab).show();
};
$(function() {
    $('.tablink-0').click(function() { selectTab('.tab-0'); });
    $('.tablink-1').click(function() { selectTab('.tab-1'); });
    $('.tablink-2').click(function() { selectTab('.tab-2'); });
});